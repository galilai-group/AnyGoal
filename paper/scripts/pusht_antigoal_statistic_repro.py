"""Reproduce joint PushT-support statistics for PushT-Antigoal."""

import os
from pathlib import Path

from pusht_avoidt_statistic_repro import main
from anygoal.paths import data_root


if __name__ == "__main__":
    dataset_root = data_root()
    main(
        query_filename=str(dataset_root / "pusht_antigoal_test.h5"),
        query_label="PushT-Antigoal",
        query_slug="push_t_antigoal",
        output_subdir="pusht_antigoal_statistic_repro",
    )
