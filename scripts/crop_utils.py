"""Shared image cropping utilities for the OCR pipeline.

Provides:
  crop_region()      — simple rectangle crop (clamped to image bounds)
  crop_polygon()     — crop to bounding box then mask outside a polygon (white fill)
  staircase_polygon() — build a tight staircase polygon string from TextLine rects
"""

from PIL import Image, ImageDraw


def crop_region(image: Image.Image, x: float, y: float, w: float, h: float) -> Image.Image:
    left   = max(0, int(x))
    top    = max(0, int(y))
    right  = min(image.width,  int(x + w))
    bottom = min(image.height, int(y + h))
    if right <= left or bottom <= top:
        return Image.new("RGB", (1, 1), color=255)
    return image.crop((left, top, right, bottom))


def crop_polygon(image: Image.Image, points_str: str) -> Image.Image:
    """Crop and mask to a staircase polygon.

    points_str: space-separated 'x,y' pairs as stored in the polygon_points CSV column.
    Returns a PIL Image with pixels outside the polygon set to white.
    """
    pts = [(int(p.split(",")[0]), int(p.split(",")[1])) for p in points_str.strip().split()]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    left   = max(0, min(xs))
    top    = max(0, min(ys))
    right  = min(image.width,  max(xs))
    bottom = min(image.height, max(ys))
    if right <= left or bottom <= top:
        return Image.new("RGB", (1, 1), color=255)
    crop = image.crop((left, top, right, bottom))
    local_pts = [(x - left, y - top) for x, y in pts]
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).polygon(local_pts, fill=255)
    white = Image.new("RGB", crop.size, 255)
    white.paste(crop, mask=mask)
    return white


def lines_from_staircase_polygon(points_str: str) -> list[tuple[int, int, int, int]]:
    """Recover (hpos, vpos, width, height) per line from a staircase polygon string.

    Inverse of staircase_polygon(). The encoding is exactly 4 points per line:
      - first 2n points: right edge top→bottom, pairs (hpos+width, vpos) then (hpos+width, vpos+height)
      - last  2n points: left edge bottom→top,  pairs (hpos, vpos+height) then (hpos, vpos) [reversed]

    Returns an empty list if points_str is malformed or encodes fewer than 1 line.
    """
    pts = [(int(p.split(",")[0]), int(p.split(",")[1])) for p in points_str.strip().split()]
    n, rem = divmod(len(pts), 4)
    if rem != 0 or n == 0:
        return []
    right_pts = pts[:2 * n]
    left_pts  = pts[2 * n:]
    lines = []
    for i in range(n):
        right_x = right_pts[2 * i][0]
        vpos    = right_pts[2 * i][1]
        height  = right_pts[2 * i + 1][1] - vpos
        left_x  = left_pts[2 * (n - 1 - i)][0]   # reversed: line i is at index n-1-i
        lines.append((left_x, vpos, right_x - left_x, height))
    return lines


def staircase_polygon(lines: list[tuple[int, int, int, int]]) -> str:
    """Compute a staircase polygon string from a list of (hpos, vpos, width, height).

    Sorts lines by vpos and traces the right edge top-to-bottom then the left edge
    bottom-to-top, creating a closed stepped polygon that tightly fits the text
    without including the adjacent column gutter.

    Returns space-separated 'x,y' pairs suitable for:
      - the polygon_points CSV column
      - ALTO <Shape><Polygon POINTS=...>
    """
    sorted_lines = sorted(lines, key=lambda l: l[1])  # sort by vpos
    right_pts = []
    for hpos, vpos, width, height in sorted_lines:
        right_pts.append((hpos + width, vpos))
        right_pts.append((hpos + width, vpos + height))
    left_pts = []
    for hpos, vpos, width, height in reversed(sorted_lines):
        left_pts.append((hpos, vpos + height))
        left_pts.append((hpos, vpos))
    pts = right_pts + left_pts
    return " ".join(f"{x},{y}" for x, y in pts)
