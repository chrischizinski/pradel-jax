"""Text covariates must be dummy-coded regardless of how pandas stores them.

Newer pandas infers text columns as StringDtype rather than object dtype. The
adapters used to branch on ``dtype == "object"``, so a StringDtype column fell
through to the numeric branch and raised

    ValueError: could not convert string to float: 'Female'

on the repo's own dipper dataset. Local runs missed it because the pinned
environment had pandas 2.0.3; scheduled CI, which installs current pandas,
caught it. These tests reproduce the failure without needing pyarrow, by using
the python-backed StringDtype.
"""

import numpy as np
import pandas as pd
import pytest

from pradel_jax.data.adapters import RMarkFormatAdapter, _is_categorical_column


@pytest.mark.parametrize(
    "series, expected",
    [
        (pd.Series(["Female", "Male"]), True),  # object dtype
        (pd.Series(["Female", "Male"], dtype="string"), True),  # StringDtype
        (pd.Series(["Female", "Male"], dtype="category"), True),
        (pd.Series([True, False]), False),  # bool casts cleanly to 0.0/1.0
        (pd.Series([1.0, 2.0]), False),
        (pd.Series([1, 2]), False),
    ],
)
def test_categorical_detection_does_not_depend_on_object_dtype(series, expected):
    assert _is_categorical_column(series) is expected


def _frame(sex_series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "individual_id": range(4),
            "ch": ["1011", "1101", "0111", "1010"],
            "sex": sex_series,
        }
    )


def test_string_dtype_covariate_is_coded_instead_of_raising():
    """The exact CI failure, at the function that raised it.

    The adapter is called directly rather than through a CSV, because a CSV
    round-trip on older pandas hands back object dtype and so cannot reproduce
    the bug at all - which is precisely why local runs stayed green while CI
    went red.
    """
    frame = _frame(pd.Series(["Female", "Male", "Female", "Male"], dtype="string"))
    assert frame["sex"].dtype == "string"  # guard: the premise of this test

    covariates = RMarkFormatAdapter().extract_covariates(frame)

    assert covariates["sex_is_categorical"] is True
    assert sorted(covariates["sex_categories"]) == ["Female", "Male"]
    sex = np.asarray(covariates["sex"])
    assert sorted(set(sex.tolist())) == [0.0, 1.0]
    assert sex[0] == sex[2] and sex[1] == sex[3] and sex[0] != sex[1]


def test_object_and_string_dtype_produce_identical_covariates():
    """How pandas happens to store the column must not change the model.

    Without this, a pandas upgrade could silently alter fitted results rather
    than failing outright, which is the harder problem to notice.
    """
    values = ["Female", "Male", "Female", "Male"]
    adapter = RMarkFormatAdapter()

    left = adapter.extract_covariates(_frame(pd.Series(values)))
    right = adapter.extract_covariates(_frame(pd.Series(values, dtype="string")))

    np.testing.assert_array_equal(
        np.asarray(left["sex"]), np.asarray(right["sex"])
    )
    assert left["sex_categories"] == right["sex_categories"]
    assert sorted(left) == sorted(right)
