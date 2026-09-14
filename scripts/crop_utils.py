"""Shared image cropping utilities for the OCR pipeline.

Provides:
  crop_region()      — simple rectangle crop (clamped to image bounds)
  crop_polygon()     — crop to bounding box then mask outside a polygon (white fill)
  staircase_polygon() — build a tight staircase polygon string from TextLine rects
  column_polygon()    — like staircase_polygon() but provably simple (no self-
                        intersections / voids), for warped or irregular columns
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

    NOTE: this uses each line's raw vpos/height, so when consecutive lines
    overlap in y, run out of order, or sit side-by-side at the same y-level
    (mastheads, split lines), the right-edge chain steps back upward and the
    polygon self-intersects — producing interior "voids" when parity-filled/
    rasterized. See column_polygon() for a drop-in replacement that returns a
    provably simple polygon (row clustering + monotone y-bands); prefer it for
    new work. This function is kept for the existing pipeline
    (extract_ssu_text.py / process_alto_labelled.py) that still depends on its
    exact 4-points-per-line encoding (see lines_from_staircase_polygon()).

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


def column_polygon(lines: list[tuple[int, int, int, int]]) -> str:
    """Build a *simple* (non-self-intersecting) column polygon from line rects.

    A robust replacement for staircase_polygon() -- see that function's
    docstring for the self-intersection/void failure mode this fixes. Not yet
    wired into the extract pipeline (extract_ssu_text.py / gt_ssu_bboxes.csv
    still call staircase_polygon()); swap it in once the input line boxes
    themselves are trustworthy -- as of 2026-07-21, per-line boxes derived
    from the ALTO XML run ~12-25px loose on the left edge (measured against
    actual page ink), so column_polygon() faithfully wraps that same
    looseness today. Tightening the box *source* (e.g. via a validated OCR
    detector, matched back to GT text/regions) is a separate, unresolved
    problem -- this function only guarantees the polygon shape is simple, not
    that the input boxes are tight.

    It traces a right edge
    top-to-bottom and a left edge bottom-to-top, tightly following per-row
    width (so it excludes the column gutter and follows page warp), but makes
    three corrections that guarantee a simple polygon — no self-intersections
    and therefore no parity-fill voids when rasterized:

      1. Row clustering. Line boxes that overlap vertically are first merged
         into a single row (its x-envelope). Real SSUs contain side-by-side
         fragments at the same y-level — a masthead's left/right/centre pieces,
         or a body line split into two String groups — and no left-edge/
         right-edge chain can represent two boxes at the same y without
         crossing. Collapsing each y-level to one row removes that entirely; a
         clean single-column SSU has exactly one line per row (unchanged).
      2. Gap-bridging monotonic y-bands. Each row owns a contiguous vertical
         band with boundaries at the midpoint between adjacent row *centres*,
         so bands tile the column with no gaps/overlaps and strictly increase
         down the page. staircase_polygon() instead used each line's raw
         vpos/height, which overlap or run out of order, making the right chain
         step back upward and cross itself — the source of the voids.
      3. Rectilinear connectors between bands rather than diagonals.

    Both chains are then monotonic in y with left_edge < right_edge at every y,
    so they cannot cross. Output format matches staircase_polygon()
    (space-separated 'x,y' pairs), now 4 points per *row*.

    Falls back to a single rectangle for one row, and "" for no lines.
    """
    if not lines:
        return ""

    # 1. Cluster lines into rows by vertical overlap. Sort by centre-y, then
    #    grow a row while the next line overlaps the row's y-envelope by more
    #    than half its height; otherwise start a new row.
    ordered = sorted(lines, key=lambda l: l[1] + l[3] / 2)
    rows: list[list[tuple[int, int, int, int]]] = [[ordered[0]]]
    r_top, r_bot = ordered[0][1], ordered[0][1] + ordered[0][3]
    for x, y, w, h in ordered[1:]:
        overlap = min(r_bot, y + h) - max(r_top, y)
        if overlap > 0.5 * h:  # same visual row
            rows[-1].append((x, y, w, h))
            r_top, r_bot = min(r_top, y), max(r_bot, y + h)
        else:                  # new row
            rows.append([(x, y, w, h)])
            r_top, r_bot = y, y + h

    # Collapse each row to (left, right, centre_y) using its x-envelope.
    row_boxes = []
    for row in rows:
        left = min(b[0] for b in row)
        right = max(b[0] + b[2] for b in row)
        top = min(b[1] for b in row)
        bot = max(b[1] + b[3] for b in row)
        row_boxes.append((left, right, top, bot, (top + bot) / 2))
    row_boxes.sort(key=lambda rb: rb[4])
    n = len(row_boxes)
    centres = [rb[4] for rb in row_boxes]

    # 2. Contiguous monotonic y-bands, one per row.
    tops: list[float] = []
    bottoms: list[float] = []
    for i, (left, right, top, bot, cy) in enumerate(row_boxes):
        tops.append(top if i == 0 else (centres[i - 1] + centres[i]) / 2)
        bottoms.append(bot if i == n - 1 else (centres[i] + centres[i + 1]) / 2)

    # 3. Right chain top->bottom, left chain bottom->up, rectilinear.
    right_pts = []
    for i, (left, right, *_rest) in enumerate(row_boxes):
        right_pts.append((right, tops[i]))
        right_pts.append((right, bottoms[i]))
    left_pts = []
    for i in range(n - 1, -1, -1):
        left = row_boxes[i][0]
        left_pts.append((left, bottoms[i]))
        left_pts.append((left, tops[i]))

    pts = right_pts + left_pts
    return " ".join(f"{int(round(x))},{int(round(y))}" for x, y in pts)
