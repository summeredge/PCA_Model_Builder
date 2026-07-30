from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


TimeWindow = tuple[pd.Timestamp, pd.Timestamp]


def ensure_disjoint_windows(
    training_windows: Sequence[TimeWindow],
    validation_windows: Sequence[TimeWindow],
) -> None:
    for train_start, train_end in training_windows:
        if train_start > train_end:
            raise ValueError("training window start must not follow its end")
        for validation_start, validation_end in validation_windows:
            if validation_start > validation_end:
                raise ValueError("validation window start must not follow its end")
            if max(train_start, validation_start) <= min(train_end, validation_end):
                raise ValueError("training and validation windows overlap")

