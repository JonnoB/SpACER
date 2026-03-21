#!/usr/bin/env python
"""Tag Spiritualist ALTO XML files with SSU identifiers."""

import logging
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "cotescore/src"))

from cotescore import assign_alto_ssu

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

INPUT_DIR = repo_root / "data/spiritualist/ocr_gt"
OUTPUT_DIR = repo_root / "data/spiritualist/ocr_gt_with_ssu"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

errors = 0
for xml_file in sorted(INPUT_DIR.glob("*.xml")):
    try:
        result = assign_alto_ssu(str(xml_file), output_path=str(OUTPUT_DIR / xml_file.name))
        logging.info("%-30s  %d SSUs", xml_file.name, len(result["ssu_to_lines"]))
    except Exception as exc:
        logging.error("%-30s  FAILED: %s", xml_file.name, exc)
        errors += 1

sys.exit(1 if errors else 0)
