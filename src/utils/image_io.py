from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def iter_image_files(directory: str | Path):
    directory = Path(directory)
    if not directory.exists():
        return
    for path in directory.rglob("*"):
        if path.is_file() and is_image_file(path):
            yield path


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def resize_square(img: Image.Image, image_size: int) -> Image.Image:
    return ImageOps.fit(img, (image_size, image_size), method=Image.Resampling.LANCZOS)
