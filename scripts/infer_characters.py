"""
Infer character-level bounding boxes from ALTO XML ground truth files.

For each word (String element), character positions are estimated by
uniformly distributing the word bounding box across its characters.
Spaces are excluded from the output.

Output: data/spiritualist/characters_inferred.parquet
Columns: char_id, page_id, char_text, x, y, w, h
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
INPUT_DIR = Path("data/spiritualist/ocr_gt")
OUTPUT_PATH = Path("data/spiritualist/characters_inferred.parquet")


def infer_characters_from_xml(xml_path: Path) -> list[dict]:
    page_id = xml_path.stem
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []
    for block_idx, block in enumerate(root.findall(".//alto:TextBlock", ALTO_NS)):
        for line_idx, line in enumerate(block.findall("alto:TextLine", ALTO_NS)):
            word_idx = 0
            for elem in line:
                tag = elem.tag.split("}")[-1]
                if tag != "String":
                    continue
                content = elem.attrib.get("CONTENT", "")
                hpos = float(elem.attrib["HPOS"])
                vpos = float(elem.attrib["VPOS"])
                width = float(elem.attrib["WIDTH"])
                height = float(elem.attrib["HEIGHT"])

                chars = [c for c in content if c != " "]
                if not chars:
                    word_idx += 1
                    continue

                char_w = width / len(chars)
                for char_idx, char in enumerate(chars):
                    char_id = f"{page_id}_b{block_idx}_l{line_idx}_w{word_idx}_c{char_idx}"
                    records.append({
                        "char_id": char_id,
                        "page_id": page_id,
                        "char_text": char,
                        "x": hpos + char_idx * char_w,
                        "y": vpos,
                        "w": char_w,
                        "h": height,
                    })
                word_idx += 1

    return records


def main():
    all_records = []
    xml_files = sorted(INPUT_DIR.glob("*.xml"))
    print(f"Processing {len(xml_files)} files...")
    for xml_path in xml_files:
        records = infer_characters_from_xml(xml_path)
        all_records.extend(records)
        print(f"  {xml_path.name}: {len(records)} characters")

    df = pd.DataFrame(all_records, columns=["char_id", "page_id", "char_text", "x", "y", "w", "h"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df):,} characters to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
