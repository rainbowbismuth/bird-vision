"""
This utility processes the paletted BMPs saved from the FFT Sprite Editor
"""

from pathlib import Path

import click
import cv2
from PIL import Image
from tqdm import tqdm

from birdvision.config import configure
from birdvision.rectangle import Rectangle


def split_unit_pieces(out_dir: Path, out_file: Path, num: int, large_only):
    img = cv2.imread(out_file.as_posix())
    thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in contours]
    rects.sort(key=lambda t: (t[1] << 16) + t[0])

    for i, (x, y, w, h) in enumerate(rects):
        if w < 3 or h < 3:
            continue
        if large_only and w < 24 and h < 24:
            continue

        r = Rectangle(x, y, w, h)
        cropped = r.crop(img)
        piece_out_dir = out_dir / f'Palette_{num}'
        piece_out_dir.mkdir(parents=True, exist_ok=True)
        piece_out_file = piece_out_dir / f'Sprite_{i}.png'
        cv2.imwrite(piece_out_file.as_posix(), cropped)


def palettes_out(src: Path, dst: Path, large_only):
    image = Image.open(src.as_posix())
    pal = image.getpalette()
    portrait = False
    for i in range(16):
        sub_pal = pal[i * 16 * 3: (i + 1) * 16 * 3]
        if not sub_pal:
            return
        if sum(sub_pal) == 0:
            continue
        if i >= 8:
            portrait = True

        out_dir = dst / src.stem
        out_sheet_dir = dst / src.stem / 'Sheets'
        out_sheet_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_sheet_dir / f'Palette_{i}.png'

        image.putpalette(sub_pal)
        image.save(out_file.as_posix(), optimize=True, format="PNG")

        if not portrait:
            split_unit_pieces(out_dir, out_file, i, large_only)


@click.command()
@click.option('--large-only/--any-size', default=False, help='Only emit sprites larger than 24 pixels in one dimension')
@click.argument('src')
@click.argument('dst')
def paletted_bmp(src, dst, large_only):
    src = Path(src)
    dst = Path(dst)
    for src_image in tqdm(list(src.glob('*.bmp'))):
        palettes_out(src_image, dst, large_only)


if __name__ == '__main__':
    configure()
    paletted_bmp()
