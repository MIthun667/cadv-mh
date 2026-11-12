import pytest

from cadvmh.data.loader import load_dataset
from cadvmh.data.preprocessing import clean_original_text, count_c1_controls, mojibake_rate


def _write(tmp_path, rows, encoding):
    path = tmp_path / "d.csv"
    header = "body,subreddit\n"
    path.write_bytes((header + "".join(f'"{t}",{label}\n' for t, label in rows)).encode(encoding))
    return {"name": "x", "path": str(path), "text_column": "body", "label_column": "subreddit", "labels": ["ocd", "adhd"],
            "label_aliases": {"0": "ocd", "1": "adhd"}}


ROWS = [("I can’t stop checking the door", 0), ("my doctor said it’s a focus issue", 1)] * 3


def test_utf8_file_decoded_as_utf8_is_clean(tmp_path):
    frame, _ = load_dataset({**_write(tmp_path, ROWS, "utf-8"), "encoding": "utf-8"})
    assert "’" in frame.text.iloc[0] and frame.attrs["mojibake_rate"] == 0


def test_wrong_single_byte_decoding_of_utf8_is_rejected_not_silently_accepted(tmp_path):
    spec = {**_write(tmp_path, ROWS, "utf-8"), "encoding": "latin-1"}  # the pre-repair configuration
    with pytest.raises(ValueError, match="mojibake"):
        load_dataset(spec)


def test_dataset_encoding_overrides_call_site_argument_and_default_is_strict_utf8(tmp_path):
    spec = _write(tmp_path, ROWS, "utf-8")
    assert load_dataset({**spec, "encoding": "utf-8"}, "latin-1")[1].rows_valid == len(ROWS)  # per-dataset key wins
    legacy = _write(tmp_path / ".", [("caf\xe9 \x9f", 0)], "latin-1")
    with pytest.raises(UnicodeDecodeError):
        load_dataset(legacy)  # no encoding given: strict UTF-8 raises instead of guessing


def test_c1_controls_are_stripped_and_counted():
    assert clean_original_text("so \x9f\x98 happy") == "so  happy"
    assert count_c1_controls(["a\x9f", "b", "\x80\x99"]) == 3
    assert mojibake_rate(["fine text", "itâ€™s"]) == 0.5
