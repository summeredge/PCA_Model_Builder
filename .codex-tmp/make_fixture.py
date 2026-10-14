import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from pca_model_builder import web, web_model_results as ui  # noqa: E402

web.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
web.RUNS_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
frame = pd.DataFrame(rng.normal(size=(240, 6)), columns=[f"TAG_{c}" for c in "ABCDEF"])
frame["TAG_A"] = frame["TAG_A"] * 40 + 120
frame.insert(0, "time", pd.date_range("2026-01-01", periods=240, freq="5min"))
uploaded = web.save_upload("audit.csv", frame.to_csv(index=False).encode())
trained = ui.train_payload({
    "file_id": uploaded["file_id"],
    "timestamp_column": "time",
    "tags": [f"TAG_{c}" for c in "ABCDEF"],
    "normal_start": frame.time.iloc[0].isoformat(),
    "normal_end": frame.time.iloc[-1].isoformat(),
    "filter_method": "none",
    "sample_interval_minutes": 5,
    "max_lag_minutes": 0,
    "lag_step_minutes": 5,
    "n_components": 3,
    "model_name": "audit",
})
print(json.dumps({"trained": trained, "file_id": uploaded["file_id"],
                  "start": frame.time.iloc[0].isoformat(), "end": frame.time.iloc[-1].isoformat()}))
