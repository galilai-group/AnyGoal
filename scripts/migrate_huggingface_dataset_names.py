"""Safely migrate the private AnyGoal dataset repository to canonical names."""

from __future__ import annotations

import json

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    HfApi,
)


REPO_ID = "HazelNam/anygoal"
REPO_TYPE = "dataset"

RENAMES = {
    "avoidt_expert_test.h5": "avoidt_test.h5",
    "stackt_v2_expert_test.h5": "stackt_test.h5",
    "push_t_antigoal_v3_expert_test.h5": "pusht_antigoal_test.h5",
    "cube_hop_expert_test.h5": "cube_hop_test.h5",
    "cube_diagonal_expert_test.h5": "cube_diagonal_test.h5",
    "manipulator_hug_box_v2_expert_test.h5": "avoid_peg_test.h5",
    "manipulator_extract_peg_v3_expert_test.h5": "extract_peg_test.h5",
    "tworoom_triangles.h5": "tworoom_route_test.h5",
}

ARTIFACTS = {
    "avoidt": ("avoidt_test.h5", 42588090, "b894d48b054c3bddd63b5fa49ecb0e1dd45a1d2151454e9d44c8639aa6a50c2b"),
    "stackt": ("stackt_test.h5", 153478895, "9aa71b759b866cf88f8829f50f559e4f17da67327434b3401ddfea8ee08e625f"),
    "pusht-antigoal": ("pusht_antigoal_test.h5", 183027627, "c589e6a536af3c7e85e580d383e96f89a99b0f81756fb37368b4953ba438aec0"),
    "cubehop": ("cube_hop_test.h5", 2941054640, "1021cbeb98f18d88fde888c076fe6f39ec252fb732e57fa858045ace8bb17b43"),
    "cubediagonal": ("cube_diagonal_test.h5", 2957466907, "ce0039b41512562cf3e9fed483a35c7e85d6394b9ca4734a4c117ca70b0bd354"),
    "insertpeg": ("manipulator_insert_peg_weak_expert_train.h5", 40032526769, "7e676e32a835e0bb7d9982ff79672d25edb476fd60adcc04e3f979657c82d98d"),
    "hug_box": ("avoid_peg_test.h5", 3511697560, "83c83532c610400734634548109abae6a5dcab13579232e75bb76590e9812644"),
    "extractpeg": ("extract_peg_test.h5", 3499555518, "afc53cffd9e3247a8e6cab38ef4f8567c897a1a70cd9733e28537518cb5932a1"),
    "tworoom_route": ("tworoom_route_test.h5", 164666413, "546d1867129663d566d3579de7b4b29401d37ee73fedbe4dca9c17ab354bb0d2"),
}

OBSOLETE = "manipulator_extract_peg_v2_expert_test.h5"


def remote_files(api: HfApi) -> dict[str, tuple[int | None, str | None]]:
    info = api.dataset_info(REPO_ID, files_metadata=True)
    if not info.private:
        raise RuntimeError(f"Refusing to modify non-private repository {REPO_ID}")
    files = {}
    for item in info.siblings:
        lfs = getattr(item, "lfs", None)
        files[item.rfilename] = (
            getattr(item, "size", None),
            getattr(lfs, "sha256", None) if lfs else None,
        )
    return files


def expected_by_filename() -> dict[str, tuple[int, str]]:
    return {filename: (size, checksum) for filename, size, checksum in ARTIFACTS.values()}


def verify_artifacts(files: dict[str, tuple[int | None, str | None]]) -> None:
    for filename, expected in expected_by_filename().items():
        actual = files.get(filename)
        if actual != expected:
            raise RuntimeError(f"Remote metadata mismatch for {filename}: {actual} != {expected}")


def main() -> None:
    api = HfApi()
    files = remote_files(api)
    expected = expected_by_filename()

    copies = []
    for source, destination in RENAMES.items():
        if destination in files:
            if files[destination] != expected[destination]:
                raise RuntimeError(f"Unexpected existing destination: {destination}")
            continue
        if source not in files or files[source] != expected[destination]:
            raise RuntimeError(f"Source does not match expected artifact: {source}")
        copies.append(CommitOperationCopy(source, destination))

    if copies:
        commit = api.create_commit(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            operations=copies,
            commit_message="Copy datasets to canonical filenames",
        )
        print(f"copy_commit={commit.oid}", flush=True)
    else:
        print("copy_commit=skipped", flush=True)

    files = remote_files(api)
    verify_artifacts(files)
    print("copy_verification=passed", flush=True)

    manifest = {
        dataset_id: {"filename": filename, "size_bytes": size, "sha256": checksum}
        for dataset_id, (filename, size, checksum) in ARTIFACTS.items()
    }
    deletions = [name for name in [*RENAMES, OBSOLETE] if name in files]
    operations = [CommitOperationDelete(name) for name in deletions]
    operations.append(
        CommitOperationAdd(
            "manifest.json",
            (json.dumps(manifest, indent=2) + "\n").encode(),
        )
    )
    commit = api.create_commit(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        operations=operations,
        commit_message="Remove superseded datasets and update manifest",
    )
    print(f"cleanup_commit={commit.oid}", flush=True)
    print(f"deleted={','.join(deletions)}", flush=True)

    files = remote_files(api)
    verify_artifacts(files)
    expected_files = {".gitattributes", "README.md", "manifest.json", *expected}
    if set(files) != expected_files:
        raise RuntimeError(
            f"Unexpected final repository files: {sorted(set(files) ^ expected_files)}"
        )
    print("final_verification=passed", flush=True)


if __name__ == "__main__":
    main()
