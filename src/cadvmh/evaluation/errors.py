from __future__ import annotations

import pandas as pd


def error_frame(sample_ids, labels, predictions, confidences) -> pd.DataFrame:
    frame = pd.DataFrame({"sample_id": sample_ids, "label": labels, "prediction": predictions, "confidence": confidences})
    return frame.loc[frame.label != frame.prediction].sort_values("confidence", ascending=False)
