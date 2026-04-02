"""
Batch OCR processor using dots.mocr via vLLM — GT crop edition.
- Reads GT bounding boxes from a CSV (columns: filename, x, y, width, height, ssu_id)
- Crops each bbox from its source image and sends the crop to the vLLM server
- Saves one parquet per page image under <output_dir>/parts/, then merges to a
  single parquet on completion
- Resumes from where it left off if interrupted (skips pages whose part already exists)

Usage:
    python scripts/mocr_crop_vllm.py \\
        --bboxes     data/spiritualist/gt_ssu_bboxes.csv \\
        --input_dir  data/spiritualist/images \\
        --output     data/spiritualist/mocr_crop_results.parquet

    python scripts/mocr_crop_vllm.py \\
        --bboxes     data/spiritualist/gt_ssu_bboxes.csv \\
        --input_dir  data/spiritualist/images \\
        --output     data/spiritualist/mocr_crop_results.parquet \\
        --prompt_mode prompt_ocr \\
        --concurrency 16 \\
        --port 8000
"""

import argparse
import asyncio
import base64
import logging
import sys
import time
from io import BytesIO
from pathlib import Path

import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.crop_utils import crop_polygon

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── prompt modes ──────────────────────────────────────────────────────────────
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

def crop_region(image: Image.Image, x: float, y: float, w: float, h: float) -> Image.Image:
    left   = max(0, int(x))
    top    = max(0, int(y))
    right  = min(image.width,  int(x + w))
    bottom = min(image.height, int(y + h))
    if right <= left or bottom <= top:
        return Image.new("RGB", (1, 1), color=255)
    return image.crop((left, top, right, bottom))


def encode_pil(image: Image.Image, max_side: int = 0) -> str:
    """Encode a PIL image to a base64 JPEG string."""
    if max_side and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=90)
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

async def ocr_crop(
    client: AsyncOpenAI,
    row: dict,
    page_image: Image.Image,
    prompt: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_side: int,
) -> dict:
    """Run OCR on a single crop. Returns the row dict augmented with ocr_text."""
    async with semaphore:
        try:
            pp = row.get("polygon_points", "")
            if pp and isinstance(pp, str) and pp.strip():
                crop = await asyncio.to_thread(crop_polygon, page_image, pp)
            else:
                crop = await asyncio.to_thread(
                    crop_region, page_image,
                    row["x"], row["y"], row["width"], row["height"],
                )
            image_b64 = await asyncio.to_thread(encode_pil, crop, max_side)
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
            return {**row, "ocr_text": response.choices[0].message.content}
        except Exception as e:
            log.error("Failed ssu_id=%s: %s", row.get("ssu_id", "?"), e)
            return {**row, "ocr_text": None}


async def process_page(
    page_df: pd.DataFrame,
    image_dir: Path,
    client: AsyncOpenAI,
    prompt: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_side: int,
) -> pd.DataFrame:
    """Process all crops for a single page image."""
    filename = page_df["filename"].iloc[0]
    img_path = image_dir / Path(filename).name
    page_image = await asyncio.to_thread(
        lambda: Image.open(img_path).convert("RGB")
    )

    tasks = [
        ocr_crop(
            client, row.to_dict(), page_image, prompt, model_name,
            semaphore, temperature, top_p, max_tokens, max_side,
        )
        for _, row in page_df.iterrows()
    ]
    results = await asyncio.gather(*tasks)
    return pd.DataFrame(results)


async def run(
    df: pd.DataFrame,
    image_dir: Path,
    parts_dir: Path,
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

    pages = df["filename"].unique()
    done_pages = {p.stem for p in parts_dir.glob("*.parquet")}
    pending_pages = [p for p in pages if Path(p).stem not in done_pages]

    log.info("%d pages total | %d already done | %d remaining",
             len(pages), len(done_pages), len(pending_pages))

    total_crops = sum(len(df[df["filename"] == p]) for p in pending_pages)
    log.info("%d crops to process across %d pages", total_crops, len(pending_pages))

    t0 = time.perf_counter()
    for i, filename in enumerate(pending_pages, 1):
        page_df = df[df["filename"] == filename].copy()
        log.info("[%d/%d] %s  (%d crops)", i, len(pending_pages), filename, len(page_df))
        try:
            result_df = await process_page(
                page_df, image_dir, client, prompt, model_name,
                semaphore, temperature, top_p, max_tokens, max_side,
            )
            part_path = parts_dir / f"{Path(filename).stem}.parquet"
            result_df.to_parquet(part_path, index=False)
            n_ok = result_df["ocr_text"].notna().sum()
            log.info("  saved %d/%d crops → %s", n_ok, len(result_df), part_path.name)
        except Exception as e:
            log.error("  Error on page %s: %s", filename, e)

    elapsed = time.perf_counter() - t0
    log.info("Finished %d pages in %.1fs", len(pending_pages), elapsed)


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batch GT-crop OCR with dots.mocr via vLLM")
    p.add_argument("--bboxes",      required=True,  help="GT bbox CSV file")
    p.add_argument("--input_dir",   required=True,  help="Directory containing page images")
    p.add_argument("--output",      required=True,  help="Output parquet file path")
    p.add_argument("--ip",          default="localhost")
    p.add_argument("--port",        default=8000, type=int)
    p.add_argument("--model_name",  default="rednote-hilab/dots.mocr")
    p.add_argument("--prompt_mode", default="prompt_ocr", choices=list(PROMPTS))
    p.add_argument("--concurrency", default=8,  type=int,
                   help="Max simultaneous requests (default: 8)")
    p.add_argument("--temperature", default=0.1, type=float)
    p.add_argument("--top_p",       default=0.9, type=float)
    p.add_argument("--max_tokens",  default=512, type=int)
    p.add_argument("--max_side",    default=0,   type=int,
                   help="Resize longest crop side before sending (0 = no resize)")
    return p.parse_args()


def main():
    args = parse_args()

    bboxes_path = Path(args.bboxes)
    image_dir   = Path(args.input_dir)
    output_path = Path(args.output)

    if not bboxes_path.exists():
        log.error("CSV not found: %s", bboxes_path)
        sys.exit(1)
    if not image_dir.exists():
        log.error("Image dir not found: %s", image_dir)
        sys.exit(1)

    df = pd.read_csv(bboxes_path)
    if "source" in df.columns:
        df = df[df["source"] == "gt"].copy()
    log.info("Loaded %d GT rows from %s", len(df), bboxes_path)

    parts_dir = output_path.parent / output_path.stem / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Image dir:  %s", image_dir)
    log.info("Output:     %s", output_path)
    log.info("Prompt:     %s", args.prompt_mode)
    log.info("Concurrency: %d", args.concurrency)

    prompt = PROMPTS[args.prompt_mode]

    asyncio.run(run(
        df=df,
        image_dir=image_dir,
        parts_dir=parts_dir,
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

    merge_parts(parts_dir, output_path)


if __name__ == "__main__":
    main()
