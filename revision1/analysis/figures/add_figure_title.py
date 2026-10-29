#!/usr/bin/env python3
"""Titled copy of a rendered figure (user, 25 Oct: titles on the co-author copies; the untitled figures stay for the
manuscript, where the title goes in the caption). Adds a white band with the title, bold Arial at the figure's axis-title
size, centred above the unchanged figure; a title wider than the figure is split over two lines.

Usage: python add_figure_title.py <figure base path without extension> <title> [--size-pt 18] [--dpi 300]
Reads <base>.png and <base>.svg (matplotlib output), writes <base>_titled.png and <base>_titled.svg.
"""
import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def split_lines(title, font, width):
    """One line if it fits, else two lines split at the word boundary that balances them best."""
    if font.getlength(title) <= width:
        return [title]
    words = title.split()
    splits = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
    first, second = min(splits, key=lambda pair: max(font.getlength(pair[0]), font.getlength(pair[1])))
    if max(font.getlength(first), font.getlength(second)) > width:
        raise SystemExit(f"title too long for the figure even on two lines: {title}")
    return [first, second]


def titled_png(source, target, title, size_px):
    image = Image.open(source).convert("RGB")
    font = ImageFont.truetype(FONT, size_px)
    lines = split_lines(title, font, image.width * 0.95)
    line_height = round(size_px * 1.25)
    band = round(size_px * 0.6) + line_height * len(lines)
    out = Image.new("RGB", (image.width, image.height + band), "white")
    out.paste(image, (0, band))
    draw = ImageDraw.Draw(out)
    for index, line in enumerate(lines):
        draw.text((image.width / 2, round(size_px * 0.45) + index * line_height), line, font=font, fill="black", anchor="mt")
    dpi = image.info.get("dpi", (300, 300))
    out.save(target, dpi=dpi)


def titled_svg(source, target, title, size_pt):
    text = source.read_text()
    root = re.search(r"<svg\b[^>]*>", text)
    tag = root.group(0)
    width, height = (float(re.search(rf'\b{name}="([0-9.]+)pt"', tag).group(1)) for name in ("width", "height"))
    font = ImageFont.truetype(FONT, 100)  # measure at 100 units, scale to the SVG's point units
    lines = split_lines(title, font, width * 0.95 * 100 / size_pt)
    line_height = size_pt * 1.25
    band = size_pt * 0.6 + line_height * len(lines)
    new_tag = re.sub(r'\bheight="[0-9.]+pt"', f'height="{height + band:.2f}pt"', tag)
    new_tag = re.sub(r'\bviewBox="[^"]*"', f'viewBox="0 0 {width:.2f} {height + band:.2f}"', new_tag)
    texts = "".join(
        f'<text x="{width / 2:.2f}" y="{size_pt * 0.45 + size_pt * 0.8 + index * line_height:.2f}" text-anchor="middle" '
        f'style="font-family: Arial; font-weight: bold; font-size: {size_pt}px; fill: #000000">{line}</text>'
        for index, line in enumerate(lines))
    body_start, body_end = root.end(), text.rindex("</svg>")
    target.write_text(text[:root.start()] + new_tag + f'<rect width="{width:.2f}" height="{height + band:.2f}" fill="#ffffff"/>'
                      + texts + f'<g transform="translate(0 {band:.2f})">' + text[body_start:body_end] + "</g></svg>\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("title")
    parser.add_argument("--size-pt", type=float, default=18)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    base = Path(args.base)
    titled_png(base.with_suffix(".png"), base.parent / f"{base.name}_titled.png", args.title, round(args.size_pt * args.dpi / 72))
    titled_svg(base.with_suffix(".svg"), base.parent / f"{base.name}_titled.svg", args.title, args.size_pt)


if __name__ == "__main__":
    main()
