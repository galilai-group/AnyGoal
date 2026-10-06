"""Reproduce joint PushT-support statistics for StackT."""

from pusht_avoidt_statistic_repro import main


if __name__ == "__main__":
    main(
        query_filename="stackt_test.h5",
        query_label="StackT",
        query_slug="stackt",
        output_subdir="pusht_stackt_statistic_repro",
    )
