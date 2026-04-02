"""
Batch OCR processor using GraniteDocling via vLLM — full page edition.
- Processes all images in an input folder
- Saves output as .json files with parsed bounding boxes, categories, text, and finish_reason
- Resumes from where it left off if interrupted
- Uses async concurrent requests to maximise GPU utilisation

Output JSON format per page:
    {
        "finish_reason": "stop" | "length",
        "truncated": false,
        "elements": [
            {"bbox": [x1, y1, x2, y2], "category": "text", "text": "..."},
            ...
        ]
    }
  Coordinates are normalised 0-999 as returned by the model.

Start the vLLM server before running:
    vllm serve ibm-granite/granite-docling-258M \\
        --host 127.0.0.1 --port 8000 \\
        --max-num-seqs 512 \\
        --max-num-batched-tokens 8192 \\
        --enable-chunked-prefill \\
        --gpu-memory-utilization 0.9

Usage:
    python scripts/granite_docling_batch_vllm.py \\
        --input_dir /path/to/images \\
        --output_dir /path/to/output

    python scripts/granite_docling_batch_vllm.py \\
        --input_dir /path/to/images \\
        --output_dir /path/to/output \\
        --concurrency 16
"""

import argparse
import asyncio
import base64
import json
import logging
import re
import sys
import time
from io import BytesIO
from pathlib import Path

from openai import AsyncOpenAI
from PIL import Image

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
PROMPT     = "Convert this page to docling."

# Matches: <loc_x1><loc_y1><loc_x2><loc_y2>[optional <category>]text
_LOC_RE = re.compile(
    r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
    r"(?:<([^>]+)>)?"
    r"(.*)"
)


# ── doctags parser ────────────────────────────────────────────────────────────

def parse_doctags(doctags: str) -> list[dict]:
    """Parse raw doctags into a list of {bbox, category, text} dicts."""
    elements = []
    for line in doctags.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LOC_RE.match(line)
        if not m:
            continue
        x1, y1, x2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        category = m.group(5) or "text"
        text     = m.group(6).strip()
        elements.append({"bbox": [x1, y1, x2, y2], "category": category, "text": text})
    return elements


# ── helpers ───────────────────────────────────────────────────────────────────

def encode_image(image_path: Path, max_side: int = 0) -> str:
    img = Image.open(image_path).convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_pending(input_dir: Path, output_dir: Path) -> list[Path]:
    all_images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    done = {p.stem for p in output_dir.glob("*.json")}
    return [p for p in all_images if p.stem not in done]


# ── async OCR ─────────────────────────────────────────────────────────────────

async def ocr_image(
    client: AsyncOpenAI,
    image_path: Path,
    model_name: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    max_side: int,
) -> tuple[Path, dict | None]:
    async with semaphore:
        try:
            image_b64 = await asyncio.to_thread(encode_image, image_path, max_side)
            response = await client.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }],
            )
            choice        = response.choices[0]
            finish_reason = choice.finish_reason
            doctags       = choice.message.content
            truncated     = finish_reason == "length"

            if truncated:
                log.warning("TRUNCATED %s (hit max_tokens)", image_path.name)

            return image_path, {
                "finish_reason": finish_reason,
                "truncated": truncated,
                "elements": parse_doctags(doctags),
            }
        except Exception as e:
            log.error("Failed %s: %s", image_path.name, e)
            return image_path, None


async def process_batch(
    pending: list[Path],
    output_dir: Path,
    model_name: str,
    ip: str,
    port: int,
    concurrency: int,
    max_tokens: int,
    max_side: int,
) -> None:
    client    = AsyncOpenAI(base_url=f"http://{ip}:{port}/v1", api_key="dummy")
    semaphore = asyncio.Semaphore(concurrency)

    total     = len(pending)
    done      = 0
    failed    = 0
    truncated = 0
    t0        = time.perf_counter()

    tasks = [
        ocr_image(client, path, model_name, semaphore, max_tokens, max_side)
        for path in pending
    ]

    for coro in asyncio.as_completed(tasks):
        image_path, result = await coro
        done += 1

        if result is not None:
            if result["truncated"]:
                truncated += 1
            out_path = output_dir / (image_path.stem + ".json")
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            elapsed = time.perf_counter() - t0
            rate    = done / elapsed
            eta     = (total - done) / rate if rate > 0 else 0
            log.info(
                "[%d/%d] ✓ %-40s  finish=%s  elements=%d  %.1fs elapsed  ETA %.0fs",
                done, total, image_path.name, result["finish_reason"],
                len(result["elements"]), elapsed, eta,
            )
        else:
            failed += 1
            log.warning("[%d/%d] ✗ %s  (skipped — see error above)", done, total, image_path.name)

    elapsed = time.perf_counter() - t0
    log.info(
        "Done. %d succeeded (%d truncated), %d failed, %.1fs total (%.2fs/image)",
        total - failed, truncated, failed, elapsed, elapsed / total if total else 0,
    )


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batch page OCR with GraniteDocling via vLLM")
    p.add_argument("--input_dir",   required=True, help="Folder of input images")
    p.add_argument("--output_dir",  required=True, help="Folder to write .json results")
    p.add_argument("--ip",          default="localhost")
    p.add_argument("--port",        default=8000, type=int)
    p.add_argument("--model_name",  default="ibm-granite/granite-docling-258M")
    p.add_argument("--concurrency", default=8, type=int,
                   help="Max simultaneous requests — pages are large so start low (default: 8)")
    p.add_argument("--max_tokens",  default=4096, type=int)
    p.add_argument("--max_side",    default=0, type=int,
                   help="Resize longest side before sending (0 = no resize)")
    return p.parse_args()


def main():
    args       = parse_args()
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        log.error("Input dir does not exist: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    pending = get_pending(input_dir, output_dir)
    if not pending:
        log.info("Nothing to do — all images already processed.")
        sys.exit(0)

    log.info("Input:       %s", input_dir)
    log.info("Output:      %s", output_dir)
    log.info("Model:       %s", args.model_name)
    log.info("Pending:     %d images", len(pending))
    log.info("Concurrency: %d", args.concurrency)

    asyncio.run(process_batch(
        pending=pending,
        output_dir=output_dir,
        model_name=args.model_name,
        ip=args.ip,
        port=args.port,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        max_side=args.max_side,
    ))


if __name__ == "__main__":
    main()
