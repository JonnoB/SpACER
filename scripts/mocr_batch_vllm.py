"""
Batch OCR processor using dots.mocr via vLLM
- Processes all images in an input folder
- Saves output as .txt files in output folder
- Resumes from where it left off if interrupted
- Uses async concurrent requests to maximise GPU utilisation

Usage:
    python batch_ocr.py --input_dir /path/to/images --output_dir /path/to/output
    python batch_ocr.py --input_dir /path/to/images --output_dir /path/to/output --concurrency 16
    python batch_ocr.py --input_dir /path/to/images --output_dir /path/to/output --prompt_mode prompt_ocr
"""

import argparse
import asyncio
import base64
import logging
import sys
import time
from io import BytesIO
from pathlib import Path

from openai import AsyncOpenAI
from PIL import Image

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── supported image extensions ───────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

# ── prompt modes (subset — add more from dots_mocr/utils/prompts.py) ─────────
PROMPTS = {
    "prompt_ocr": "Extract the text content from this image.",
    "prompt_layout_all_en": (
        "Please output the layout information from the PDF image, including each layout "
        "element's bbox, its category, and the corresponding text content within the bbox.\n"
        "1. Bbox format: [x1, y1, x2, y2]\n"
        "2. Layout Categories: ['Caption','Footnote','Formula','List-item','Page-footer',"
        "'Page-header','Picture','Section-header','Table','Text','Title']\n"
        "3. Text Extraction & Formatting Rules:\n"
        "   - Picture: omit text field.\n"
        "   - Formula: LaTeX.\n"
        "   - Table: HTML.\n"
        "   - All others: Markdown.\n"
        "4. All layout elements sorted by reading order.\n"
        "5. Final output must be a single JSON object."
    ),
    "prompt_scene_spotting": "Detect and recognize the text in the image.",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def encode_image(image_path: Path, max_side: int = 0) -> str:
    """Open image, optionally resize, return base64 JPEG string."""
    img = Image.open(image_path).convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def get_pending(input_dir: Path, output_dir: Path) -> list[Path]:
    """Return image paths that have not yet been processed."""
    all_images = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    done = {p.stem for p in output_dir.glob("*.txt")}
    pending = [p for p in all_images if p.stem not in done]
    return pending


# ── async OCR ─────────────────────────────────────────────────────────────────

async def ocr_image(
    client: AsyncOpenAI,
    image_path: Path,
    prompt: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_side: int,
) -> tuple[Path, str | None]:
    """Run OCR on a single image. Returns (path, text_or_None)."""
    async with semaphore:
        try:
            image_b64 = await asyncio.to_thread(encode_image, image_path, max_side)
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return image_path, response.choices[0].message.content
        except Exception as e:
            log.error("Failed %s: %s", image_path.name, e)
            return image_path, None


async def process_batch(
    pending: list[Path],
    output_dir: Path,
    model_name: str,
    prompt: str,
    ip: str,
    port: int,
    concurrency: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_side: int,
) -> None:
    client = AsyncOpenAI(base_url=f"http://{ip}:{port}/v1", api_key="dummy")
    semaphore = asyncio.Semaphore(concurrency)

    total = len(pending)
    done = 0
    failed = 0
    t0 = time.perf_counter()

    tasks = [
        ocr_image(
            client, path, prompt, model_name, semaphore,
            temperature, top_p, max_tokens, max_side,
        )
        for path in pending
    ]

    for coro in asyncio.as_completed(tasks):
        image_path, text = await coro
        done += 1

        if text is not None:
            out_path = output_dir / (image_path.stem + ".txt")
            out_path.write_text(text, encoding="utf-8")
            elapsed = time.perf_counter() - t0
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0
            log.info(
                "[%d/%d] ✓ %-40s  %.1fs elapsed  ETA %.0fs",
                done, total, image_path.name, elapsed, eta,
            )
        else:
            failed += 1
            log.warning("[%d/%d] ✗ %s  (skipped — see error above)", done, total, image_path.name)

    elapsed = time.perf_counter() - t0
    log.info(
        "Done. %d succeeded, %d failed, %.1fs total (%.2fs/image)",
        total - failed, failed, elapsed, elapsed / total if total else 0,
    )


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batch OCR with dots.mocr via vLLM")
    p.add_argument("--input_dir",   required=True,  help="Folder of input images")
    p.add_argument("--output_dir",  required=True,  help="Folder to write .txt results")
    p.add_argument("--ip",          default="localhost")
    p.add_argument("--port",        default=8000, type=int)
    p.add_argument("--model_name",  default="rednote-hilab/dots.mocr")
    p.add_argument("--prompt_mode", default="prompt_ocr", choices=list(PROMPTS))
    p.add_argument("--concurrency", default=8,  type=int,
                   help="Max simultaneous requests (start at 8, raise if GPU <80%%)")
    p.add_argument("--temperature", default=0.1, type=float)
    p.add_argument("--top_p",       default=0.9, type=float)
    p.add_argument("--max_tokens",  default=4096, type=int)
    p.add_argument("--max_side",    default=0,   type=int,
                   help="Resize longest side to this before sending (0 = no resize)")
    return p.parse_args()


def main():
    args = parse_args()
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

    log.info("Input:   %s", input_dir)
    log.info("Output:  %s", output_dir)
    log.info("Pending: %d images", len(pending))
    log.info("Prompt:  %s", args.prompt_mode)
    log.info("Concurrency: %d", args.concurrency)

    prompt = PROMPTS[args.prompt_mode]

    asyncio.run(process_batch(
        pending=pending,
        output_dir=output_dir,
        model_name=args.model_name,
        prompt=prompt,
        ip=args.ip,
        port=args.port,
        concurrency=args.concurrency,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_side=args.max_side,
    ))


if __name__ == "__main__":
    main()