import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.image_io import iter_image_files, load_rgb_image, resize_square
from src.utils.paths import ensure_dir, project_path
from src.utils.seed import set_seed


def split_records(records: list[dict], val_size: float, test_size: float, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_label: dict[int, list[dict]] = {}
    for record in records:
        by_label.setdefault(int(record["label"]), []).append(record)

    splits = {"train": [], "val": [], "test": []}
    for label_records in by_label.values():
        rng.shuffle(label_records)
        total = len(label_records)
        test_count = int(total * test_size)
        val_count = int(total * val_size)

        test_records = label_records[:test_count]
        val_records = label_records[test_count : test_count + val_count]
        train_records = label_records[test_count + val_count :]

        splits["train"].extend(train_records)
        splits["val"].extend(val_records)
        splits["test"].extend(test_records)

    for split_records_ in splits.values():
        rng.shuffle(split_records_)
    return splits


def process_class(
    source_dir: Path,
    output_dir: Path,
    class_name: str,
    label: int,
    image_size: int,
    max_images: int | None,
) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    skipped: list[dict] = []
    target_dir = ensure_dir(output_dir / class_name)
    image_paths = list(iter_image_files(source_dir) or [])
    if max_images is not None:
        image_paths = image_paths[:max_images]

    for idx, path in enumerate(tqdm(image_paths, desc=f"Processing {class_name}", unit="img"), start=1):
        try:
            image = load_rgb_image(path)
            original_width, original_height = image.size
            image = resize_square(image, image_size)
            output_path = target_dir / f"{class_name}_{idx:06d}.jpg"
            image.save(output_path, format="JPEG", quality=95)
            records.append(
                {
                    "filepath": str(output_path.resolve()),
                    "label": label,
                    "class_name": class_name,
                    "original_path": str(path.resolve()),
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_size": image_size,
                }
            )
        except Exception as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
    return records, skipped


def write_split_csvs(splits: dict[str, list[dict]], splits_dir: Path) -> None:
    ensure_dir(splits_dir)
    columns = [
        "filepath",
        "label",
        "class_name",
        "original_path",
        "original_width",
        "original_height",
        "image_size",
    ]
    for split_name, split_records_ in splits.items():
        frame = pd.DataFrame(split_records_, columns=columns)
        frame.to_csv(splits_dir / f"{split_name}.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare real/fake image dataset for training.")
    parser.add_argument("--real-dir", type=Path, default=project_path("data", "raw", "real"))
    parser.add_argument("--fake-dir", type=Path, default=project_path("data", "raw", "fake"))
    parser.add_argument("--output-dir", type=Path, default=project_path("data", "processed"))
    parser.add_argument("--splits-dir", type=Path, default=project_path("data", "splits"))
    parser.add_argument("--image-size", type=int, default=128, choices=[64, 128, 256])
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(args.splits_dir)

    real_records, real_skipped = process_class(
        source_dir=args.real_dir,
        output_dir=args.output_dir,
        class_name="real",
        label=0,
        image_size=args.image_size,
        max_images=args.max_per_class,
    )
    fake_records, fake_skipped = process_class(
        source_dir=args.fake_dir,
        output_dir=args.output_dir,
        class_name="fake",
        label=1,
        image_size=args.image_size,
        max_images=args.max_per_class,
    )

    records = real_records + fake_records
    skipped = real_skipped + fake_skipped

    if not records:
        print("No images were found. Add images to data/raw/real and data/raw/fake, or pass --real-dir and --fake-dir.")
        write_split_csvs({"train": [], "val": [], "test": []}, args.splits_dir)
        return 0

    splits = split_records(records, val_size=args.val_size, test_size=args.test_size, seed=args.seed)
    write_split_csvs(splits, args.splits_dir)

    metadata = {
        "image_size": args.image_size,
        "real_images": len(real_records),
        "fake_images": len(fake_records),
        "total_images": len(records),
        "splits": {name: len(items) for name, items in splits.items()},
        "skipped_count": len(skipped),
        "skipped": skipped[:100],
    }
    with open(args.splits_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(json.dumps(metadata, indent=2))
    print(f"Prepared dataset saved to: {args.output_dir}")
    print(f"Split CSV files saved to: {args.splits_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
