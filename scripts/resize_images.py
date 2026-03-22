"""
Batch resize images from one DPI to another.

Usage:
    python scripts/resize_images.py \\
        --input-dir /teamspace/lightning_storage/the_spiritualist/spiritualist_images \\
        --output-dir /teamspace/lightning_storage/the_spiritualist/spiritualist_images_120dpi \\
        --source-dpi 300 \\
        --target-dpi 120
"""

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description="Batch resize images between DPI targets.")
    parser.add_argument("--input-dir", required=True, help="Directory of source images")
    parser.add_argument("--output-dir", required=True, help="Directory to write resized images")
    parser.add_argument("--source-dpi", type=float, default=300)
    parser.add_argument("--target-dpi", type=float, default=120)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scale = args.target_dpi / args.source_dpi
    image_files = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"{len(image_files)} images | scale {scale:.3f} ({args.source_dpi} -> {args.target_dpi} DPI)")

    for src in tqdm(image_files, desc="Resizing"):
        dst = output_dir / src.name
        if dst.exists():
            continue
        img = Image.open(src)
        new_size = (round(img.width * scale), round(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
        img.save(dst, dpi=(args.target_dpi, args.target_dpi))

    print(f"Done -> {output_dir}")


if __name__ == "__main__":
    main()
