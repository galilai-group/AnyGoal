"""Upload exactly one approved Manipulator test dataset to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
from huggingface_hub import HfApi


REPO_ID = "HazelNam/anygoal"
DATA_ROOT = Path("/grid/klindt/home/nam/.stable_worldmodel/datasets")
DATASETS = {
    "hug_box": "avoid_peg_test.h5",
    "extractpeg": "extract_peg_test.h5",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id", choices=DATASETS)
    args = parser.parse_args()
    path = DATA_ROOT / DATASETS[args.dataset_id]
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as dataset:
        for key in ("pixels", "ep_offset", "ep_len"):
            if key not in dataset:
                raise KeyError(f"{path} is missing {key!r}")
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    print(f"Uploading {args.dataset_id}: {path}", flush=True)
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path.name,
        repo_id=REPO_ID,
        repo_type="dataset",
    )


if __name__ == "__main__":
    main()
