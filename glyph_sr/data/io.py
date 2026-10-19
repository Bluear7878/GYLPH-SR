"""JSONL and image-folder input shared by data preparation commands."""

import json
from pathlib import Path


def load_metadata(path):
    """Yield JSON objects, reporting the source line for malformed metadata."""
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield row


def write_record(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class ImageFolderDataset:
    """Discover images without importing the model or PyTorch stack."""

    def __init__(self, image_dir, exts=("jpg", "jpeg", "png", "bmp")):
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise NotADirectoryError(f"Image directory not found: {image_dir}")
        suffixes = {f".{extension.lower()}" for extension in exts}
        self.paths = sorted(
            path
            for path in self.image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image

        path = self.paths[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        return {"image": image, "path": str(path)}


def collate_images(batch):
    """Keep every image; PIL objects cannot use the default tensor collator."""
    return batch


def make_image_loader(image_dir, batch_size=1, shuffle=False, num_workers=4):
    from torch.utils.data import DataLoader

    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    return DataLoader(
        ImageFolderDataset(image_dir),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_images,
    )
