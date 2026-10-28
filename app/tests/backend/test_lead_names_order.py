"""Lead names and column order at the API boundary (stage 17). Synthetic inputs only; no model is run, and no PTB-XL
waveform or fold-10 file is read."""
import io

import numpy as np
import pandas as pd
import pytest

from app.backend import schema
from app.backend.pipeline import demo

from .test_schema_integrity import csv_text, refused, request


def test_column_order_is_mapped_by_name_not_position():
    ref = schema.parse_request(request())
    same_signals_new_order = pd.read_csv(io.StringIO(csv_text()))[["V1", "I", "III"]].to_csv(index=False)
    perm = schema.parse_request(request(observed_leads=["V1", "I", "III"], csv=same_signals_new_order))
    assert perm.observed == ref.observed == ["I", "III", "V1"]                       # canonical order
    assert np.array_equal(perm.x, ref.x)                                             # same model input
    assert np.array_equal(perm.x[demo.LEAD_INDEX["V1"]], ref.x[demo.LEAD_INDEX["V1"]])


def test_lower_case_header_names_are_read_as_the_same_leads():
    ref = schema.parse_request(request())
    low = schema.parse_request(request(csv=csv_text(["I", "III", "V1"]).replace("III", "iii", 1)))
    assert low.observed == ref.observed and np.array_equal(low.x, ref.x)


@pytest.mark.parametrize("header", ["I,III,X9", "I,I,V1", "I,III,V1,V1"])
def test_unknown_or_duplicate_csv_lead_names_are_refused(header):
    body = "".join(f"{0.5 * np.sin(k / 40):.4f},{0.4 * np.cos(k / 40):.4f},{0.3 * np.sin(k / 30):.4f}"
                   + (",0.1" if header.count(",") == 3 else "") + "\n" for k in range(5000))
    refused(request(csv=header + "\n" + body), "INVALID_CSV")
