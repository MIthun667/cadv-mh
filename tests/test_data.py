import pandas as pd
import pytest

from cadvmh.data.validation import validate_frame


def test_valid_schema_and_missing_text():
    frame = pd.DataFrame({"text": ["hello", None, "  "], "status": ["Normal", "Normal", "Normal"]})
    valid, summary = validate_frame(frame, "text", "status", ["normal"], {"Normal": "normal"})
    assert len(valid) == 1
    assert summary.missing_text == 1
    assert summary.empty_text == 1


def test_invalid_label_fails():
    with pytest.raises(ValueError, match="Invalid labels"):
        validate_frame(pd.DataFrame({"text": ["x"], "label": ["unknown"]}), "text", "label", ["normal"])
