"""Send Spiritualist OCR crops to Doubleword asynchronously (live endpoint).

The numbered 01..04 scripts use Doubleword's *batch* API (upload JSONL, create a
job, poll, download). This is the live alternative: it reads the same crop
manifest and fires each page at the /v1/chat/completions endpoint concurrently
via AsyncOpenAI, bounded by a semaphore, writing responses to a JSONL as they
return. Use it for interactive / smaller runs where waiting on the 24h batch
window isn't worth it.

The output JSONL mirrors the batch download shape (one object per line with
`custom_id` and `response.body`), so downstream scripts like 04_build_page_jsons
can consume it. Re-running skips any custom_id already present in the output,
so an interrupted run resumes cleanly.

Usage:
    uv run scripts/ocr_batch/send_async.py [OPTIONS]

Options:
    --manifest PATH       Path to manifest CSV (default: data/data_crops/manifest.csv)
    --crops-dir PATH      Path to crop images (default: data/data_crops)
    --output PATH         Output JSONL (default: output/ocr_batches/async_results.jsonl)
    --model MODEL         Model preset: olmocr, lightonocr, deepseek-ocr (default: olmocr)
    --concurrency N       Max in-flight requests (default: 8)
    --max-long-side N     Override the model preset's resize cap
    --no-resize           Send crops at their encoded size (no resize cap)
    --temperature FLOAT   Sampling temperature injected into every request
    --limit N             Only process the first N manifest rows (dry run)

Requires:
    DOUBLEWORD_API_KEY set in the .env file.

Examples:
    uv run scripts/ocr_batch/send_async.py --limit 10
    uv run scripts/ocr_batch/send_async.py --model lightonocr --concurrency 16
"""
# /// script
# requires-python = ">=3.9"
# dependencies = ["openai", "python-dotenv", "pandas", "Pillow", "tqdm"]
# ///

import argparse
import asyncio
import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image
from tqdm import tqdm

BASE_URL = "https://api.doubleword.ai/v1"

# Kept in sync with scripts/ocr_batch/01_build_batch_files.py so the live and
# batch paths send byte-identical requests for the same crop.
MODEL_PRESETS = {
    "olmocr": {
        "model_id": "allenai/olmOCR-2-7B-1025-FP8",
        "prompt": (
            "Attached is one page of a document that you must process. Just return the "
            "plain text representation of this document as if you were reading it "
            "naturally. Convert equations to LateX and tables to HTML.\n"
            "If there are any figures or charts, label them with the following markdown "
            "syntax ![Alt text describing the contents of the figure]"
            "(page_startx_starty_width_height.png)"
        ),
        "max_long_side": 1288,
        "extra_body": {"max_tokens": 4096},
    },
    "lightonocr": {
        "model_id": "lightonai/LightOnOCR-2-1B-bbox-soup",
        "prompt": None,
        "max_long_side": 1540,
        "extra_body": {"max_tokens": 4096, "temperature": 0.2, "top_p": 0.9},
    },
    "deepseek-ocr": {
        "model_id": "deepseek-ai/DeepSeek-OCR-2",
        "prompt": "Free OCR.",
        "max_long_side": 1288,
        "extra_body": {"max_tokens": 4096},
    },
}

MAX_RETRIES = 5


def encode_crop(crop_path: Path, max_long_side: Optional[int]) -> bytes:
    """Open an image, resize if the longest side exceeds *max_long_side*, return PNG bytes."""
    img = Image.open(crop_path)
    w, h = img.size
    if max_long_side is not None and max(w, h) > max_long_side:
        scale = max_long_side / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_messages(image_bytes: bytes, preset: dict) -> list:
    """Build the chat `messages` list for one crop (prompt text + image data URL)."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    content = []
    if preset["prompt"] is not None:
        content.append({"type": "text", "text": preset["prompt"]})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return [{"role": "user", "content": content}]


async def send_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    custom_id: str,
    image_bytes: bytes,
    preset: dict,
) -> dict:
    """Send one crop and return a batch-download-shaped result dict.

    Retries transient errors with exponential backoff. On final failure the
    returned dict carries an `error` key instead of a `response`.
    """
    messages = build_messages(image_bytes, preset)
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=preset["model_id"],
                    messages=messages,
                    **preset["extra_body"],
                )
                return {
                    "custom_id": custom_id,
                    "response": {"status_code": 200, "body": resp.model_dump()},
                }
            except Exception as exc:  # noqa: BLE001 -- record and move on
                if attempt == MAX_RETRIES - 1:
                    return {"custom_id": custom_id, "error": repr(exc)}
                await asyncio.sleep(2**attempt)


async def run(args) -> None:
    """Encode crops, dispatch them concurrently, and stream results to the output JSONL."""
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(project_root / ".env")
    api_key = os.environ.get("DOUBLEWORD_API_KEY")
    if not api_key:
        print("Error: DOUBLEWORD_API_KEY not set (check .env)", file=sys.stderr)
        sys.exit(1)

    preset = MODEL_PRESETS[args.model]
    if args.temperature is not None:
        preset = {**preset, "extra_body": {**preset["extra_body"], "temperature": args.temperature}}
    max_long_side = None if args.no_resize else (args.max_long_side or preset["max_long_side"])

    manifest_path = project_root / args.manifest
    crops_dir = project_root / args.crops_dir
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip any custom_id already written to the output file.
    done_ids = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["custom_id"])
        if done_ids:
            print(f"Resuming: {len(done_ids)} result(s) already in {output_path}, skipping those.")

    df = pd.read_csv(manifest_path)
    if args.limit:
        df = df.head(args.limit)
    has_custom_id_col = "custom_id" in df.columns

    # Encode up front (CPU-bound), caching per crop so a manifest that references
    # the same image many times (e.g. self-consistency runs) encodes it once.
    encode_cache: dict = {}
    jobs = []  # (custom_id, image_bytes)
    n_missing = 0
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Encoding crops"):
        crop_file = crops_dir / Path(row.crop_path).name
        custom_id = row.custom_id if has_custom_id_col else crop_file.stem
        if custom_id in done_ids:
            continue
        if not crop_file.exists():
            tqdm.write(f"Warning: crop not found, skipping: {crop_file}")
            n_missing += 1
            continue
        if crop_file not in encode_cache:
            encode_cache[crop_file] = encode_crop(crop_file, max_long_side)
        jobs.append((custom_id, encode_cache[crop_file]))

    if not jobs:
        print(f"Nothing to send ({n_missing} missing, {len(done_ids)} already done).")
        return

    client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL)
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(send_one(client, sem, cid, img, preset))
        for cid, img in jobs
    ]

    n_ok, n_err = 0, 0
    # Append as each request completes so an interrupt keeps finished work.
    with open(output_path, "a") as out:
        for coro in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Sending to Doubleword"
        ):
            result = await coro
            out.write(json.dumps(result) + "\n")
            out.flush()
            if "error" in result:
                n_err += 1
                tqdm.write(f"  error {result['custom_id']}: {result['error']}")
            else:
                n_ok += 1

    await client.close()
    print(f"Done: {n_ok} ok, {n_err} errors, {n_missing} missing. -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/data_crops/manifest.csv")
    parser.add_argument("--crops-dir", default="data/data_crops")
    parser.add_argument("--output", default="output/ocr_batches/async_results.jsonl")
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="olmocr")
    parser.add_argument("--concurrency", type=int, default=8, help="Max in-flight requests")
    parser.add_argument("--max-long-side", type=int, help="Override the model preset's resize cap")
    parser.add_argument("--no-resize", action="store_true", help="Send crops at their encoded size")
    parser.add_argument("--temperature", type=float, help="Sampling temperature for every request")
    parser.add_argument("--limit", type=int, help="Only process the first N manifest rows (dry run)")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
