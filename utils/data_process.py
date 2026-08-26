"""SAT dataset preparation for AIVE evaluation.

Downloads the SAT (Spatial Aptitude Training) benchmark from HuggingFace
(``array/SAT``), persists the RGB views as PNG files, and writes a
``{split}.json`` file with one normalised record per question.

The output layout is consumed directly by :class:`pipelines.AIVE_baseline.PipelineBase`:

    data/
    └── val/
        ├── val.json
        ├── image_0_0.png
        ├── image_0_1.png
        └── ...

Usage::

    python utils/data_process.py --split val
    python utils/data_process.py --split test
"""

from __future__ import annotations

import argparse
import io
import json
import os

from PIL import Image


def _to_pil_image(value: object) -> Image.Image:
    """Coerce a raw ``image_bytes`` value into a ``PIL.Image``.

    The ``array/SAT`` dataset decodes image bytes into ``PIL.Image`` objects,
    but some cached versions expose raw bytes instead. This helper handles both.
    """
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    raise TypeError(f"Unsupported image_bytes value: {type(value)!r}")


def _normalise_for_split(image: Image.Image, split: str) -> Image.Image:
    """Apply the split-specific image preprocessing.

    For the ``test`` split the view is center-cropped to a square and resized
    to 512 x 512 (matching the paper's evaluation protocol).
    """
    if split != "test":
        return image

    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    image = image.crop((left, top, left + min_dim, top + min_dim))
    return image.resize((512, 512), resample=Image.Resampling.BICUBIC)


def process(
    output_dir: str = "./data",
    split: str = "val",
    max_per_type: int | None = None,
) -> None:
    """Download SAT and write normalised JSON + image files.

    Parameters
    ----------
    output_dir : str
        Root directory under which ``{split}/`` is created.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    max_per_type : int, optional
        Cap the number of examples kept per question type.
    """
    from datasets import load_dataset

    print(f"[INFO] Loading SAT ({split} split) from HuggingFace ...")
    dataset = load_dataset("array/SAT")

    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    records: list[dict] = []
    type_counts: dict[str, int] = {}

    for i, example in enumerate(dataset[split]):
        question_type = str(example["question_type"])

        if max_per_type is not None and type_counts.get(question_type, 0) >= max_per_type:
            continue
        type_counts[question_type] = type_counts.get(question_type, 0) + 1

        img_paths: list[str] = []
        for idx, image_value in enumerate(example["image_bytes"]):
            image = _normalise_for_split(_to_pil_image(image_value), split)
            filename = os.path.join(split_dir, f"image_{i}_{idx}.png")
            image.save(filename, format="PNG")
            img_paths.append(filename)

        records.append(
            {
                "database_idx": str(i),
                "question_type": question_type,
                "question": example["question"],
                "answer_choices": list(example["answers"]),
                "correct_answer": example["correct_answer"],
                "img_paths": img_paths,
            }
        )

        if (i + 1) % 100 == 0:
            print(f"[INFO] Processed {i + 1}/{len(dataset[split])} examples ...")

    json_path = os.path.join(split_dir, f"{split}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)

    print(f"[INFO] Saved {len(records)} records to {json_path}")
    for qtype, count in sorted(type_counts.items()):
        print(f"        {qtype}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare the SAT dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Root output directory (creates {output_dir}/{split}/).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to prepare.",
    )
    parser.add_argument(
        "--max_per_type",
        type=int,
        default=None,
        help="Maximum examples per question type (None = all).",
    )
    args = parser.parse_args()

    process(output_dir=args.output_dir, split=args.split, max_per_type=args.max_per_type)


if __name__ == "__main__":
    main()
