"""
Batch OCR processor using olmOCR via DeepInfra API — full page edition.
- Reads a list of page images from a directory (or a CSV with a 'filename' column)
- Sends each full page image to the DeepInfra olmOCR API
- Saves one parquet per page under <output_dir>/parts/, then merges to a
  single parquet on completion
- Resumes from where it left off if interrupted

Usage:
    python scripts/olmocr_page_deepinfra.py \\
        --input_dir  data/spiritualist/images \\
        --output     data/spiritualist/olmocr_page_results.parquet

    python scripts/olmocr_page_deepinfra.py \\
        --input_dir  data/spiritualist/images \\
        --output     data/spiritualist/olmocr_page_results.parquet \\
        --test
"""

import argparse
import asyncio
import base64
import logging
import sys
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
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

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_MODEL      = "allenai/olmOCR-2-7B-1025"
IMAGE_EXTENSIONS   = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_api_key(env_path: Path) -> str:
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("DEEPINFRA_API_TOKEN"):
            return line.split("=", 1)[1].strip()
    raise ValueError(f"DEEPINFRA_API_TOKEN not found in {env_path}")


def encode_image(image_path: Path, max_side: int = 0) -> str:
    image = Image.open(image_path).convert("RGB")
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def merge_parts(parts_dir: Path, output_path: Path) -> None:
    parts = sorted(parts_dir.glob("*.parquet"))
    if not parts:
        log.warning("No part files found in %s", parts_dir)
        return
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(output_path, index=False)
    log.info("Merged %d parts → %s  (%d rows)", len(parts), output_path, len(df))


# ── async OCR ─────────────────────────────────────────────────────────────────

async def ocr_page(
    client: AsyncOpenAI,
    image_path: Path,
    model_name: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    max_side: int,
) -> dict:
    async with semaphore:
        try:
            image_b64 = await asyncio.to_thread(encode_image, image_path, max_side)
            response = await client.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an OCR text extraction algorithm. Output only the raw text from the image with no commentary, labels, or explanation.",
                    },
                    {
                        "role": "user",
                        "content": [{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        }],
                    },
                ],
            )
            return {"filename": image_path.name, "ocr_text": response.choices[0].message.content}
        except Exception as e:
            log.error("Failed %s: %s", image_path.name, e)
            return {"filename": image_path.name, "ocr_text": None}


async def run(
    image_paths: list[Path],
    parts_dir: Path,
    model_name: str,
    api_key: str,
    concurrency: int,
    max_tokens: int,
    max_side: int,
) -> None:
    client    = AsyncOpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL)
    semaphore = asyncio.Semaphore(concurrency)

    done       = {p.stem for p in parts_dir.glob("*.parquet")}
    pending    = [p for p in image_paths if p.stem not in done]

    log.info("%d images total | %d already done | %d remaining",
             len(image_paths), len(done), len(pending))

    t0 = time.perf_counter()
    for i, image_path in enumerate(pending, 1):
        log.info("[%d/%d] %s", i, len(pending), image_path.name)
        try:
            result = await ocr_page(client, image_path, model_name, semaphore, max_tokens, max_side)
            part_path = parts_dir / f"{image_path.stem}.parquet"
            pd.DataFrame([result]).to_parquet(part_path, index=False)
            status = "ok" if result["ocr_text"] else "FAILED"
            log.info("  [%s] → %s", status, part_path.name)
        except Exception as e:
            log.error("  Error on %s: %s", image_path.name, e)

    elapsed = time.perf_counter() - t0
    log.info("Finished %d pages in %.1fs", len(pending), elapsed)


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batch full-page OCR with olmOCR via DeepInfra")
    p.add_argument("--input_dir",   required=True,  help="Directory containing page images")
    p.add_argument("--output",      required=True,  help="Output parquet file path")
    p.add_argument("--env",         default="scripts/.env", help="Path to .env file with DEEPINFRA_API_TOKEN")
    p.add_argument("--model_name",  default=DEFAULT_MODEL)
    p.add_argument("--concurrency", default=4,    type=int)
    p.add_argument("--max_tokens",  default=4092, type=int)
    p.add_argument("--max_side",    default=0,    type=int,
                   help="Resize longest side before sending (0 = no resize)")
    p.add_argument("--test",        action="store_true",
                   help="Test mode: process only the first image")
    return p.parse_args()


def main():
    args = parse_args()

    image_dir   = Path(args.input_dir)
    output_path = Path(args.output)
    env_path    = Path(args.env)

    for path, label in [(image_dir, "Image dir"), (env_path, ".env")]:
        if not path.exists():
            log.error("%s not found: %s", label, path)
            sys.exit(1)

    api_key = load_api_key(env_path)

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        log.error("No images found in %s", image_dir)
        sys.exit(1)

    if args.test:
        image_paths = image_paths[:1]
        log.info("TEST MODE: restricted to 1 image (%s)", image_paths[0].name)

    log.info("Found %d images in %s", len(image_paths), image_dir)
    log.info("Output:      %s", output_path)
    log.info("Model:       %s", args.model_name)
    log.info("Concurrency: %d", args.concurrency)

    parts_dir = output_path.parent / output_path.stem / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(run(
        image_paths=image_paths,
        parts_dir=parts_dir,
        model_name=args.model_name,
        api_key=api_key,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        max_side=args.max_side,
    ))

    merge_parts(parts_dir, output_path)


if __name__ == "__main__":
    main()
