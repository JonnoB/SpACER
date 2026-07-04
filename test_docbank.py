import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    from datasets import load_dataset
    import itertools, json


    return itertools, load_dataset


@app.cell
def _(itertools, load_dataset):

    ds = load_dataset("maveriq/DocBank", split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=5000)
    subset = list(itertools.islice(ds, 300))


    return (subset,)


@app.cell
def _(subset):
    row0 = subset[20]
    return (row0,)


@app.cell
def _(row0):
    img_field = row0['image']
    print(type(img_field))
    if isinstance(img_field, dict):
        print(img_field.keys())
    return


@app.cell
def _(row0):
    import io
    from PIL import Image

    img_field2 = row0['image']

    if isinstance(img_field2, dict):
        img = Image.open(io.BytesIO(img_field2['bytes']))
    else:
        img = img_field2 # already a PIL.Image

    print(img.size)
    img
    return Image, io


@app.cell
def _(Image, io, subset):
    import hashlib, os

    import pandas as pd

    img_dir = "data/docbank/images"
    os.makedirs(img_dir, exist_ok=True)

    records = []
    seen = set()
    for row in subset:
        img_bytes = row["image"]["bytes"]
        image_id = hashlib.sha1(img_bytes).hexdigest()[:16]

        if image_id not in seen:
            Image.open(io.BytesIO(img_bytes)).save(f"{img_dir}/{image_id}.png")
            seen.add(image_id)

        records.append({
            "image_id": image_id,
            "token": row["token"],
            "bounding_box": row["bounding_box"][0],  # flatten the length-1 list
            "color": row["color"][0],
            "font": row["font"],
            "label": row["label"],
        })

    os.makedirs("data/docbank/gt", exist_ok=True)
    pd.DataFrame(records).to_parquet("data/docbank/gt/docbank_gt.parquet")

    return pd, records


@app.cell
def _(pd, records):
    pd.DataFrame(records)
    return


if __name__ == "__main__":
    app.run()
