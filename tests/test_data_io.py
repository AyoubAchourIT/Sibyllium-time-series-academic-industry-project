import pandas as pd

from src.data.io import load_excel_ohlcv


def test_load_excel_ohlcv_normalizes_columns_and_uses_first_sheet(tmp_path):
    path = tmp_path / "sample.xlsx"

    first_sheet = pd.DataFrame(
        {
            "DATE": ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02"],
            "Open": [3.0, 1.0, 2.0, 20.0],
            "HIGH": [3.5, 1.5, 2.5, 20.5],
            "low": [2.5, 0.5, 1.5, 19.5],
            "Close": [3.2, 1.2, 2.2, 20.2],
            "VoLuMe": [300, 100, 200, 2000],
            "extra_col": ["x", "y", "z", "dup"],
        }
    )
    second_sheet = pd.DataFrame(
        {
            "date": ["2020-01-01"],
            "open": [999],
            "high": [999],
            "low": [999],
            "close": [999],
            "volume": [999],
        }
    )

    with pd.ExcelWriter(path) as writer:
        first_sheet.to_excel(writer, index=False, sheet_name="first")
        second_sheet.to_excel(writer, index=False, sheet_name="second")

    df = load_excel_ohlcv(path)

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "date"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tolist() == list(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))

    # Duplicate date keeps the last row after sorting.
    jan2 = pd.Timestamp("2024-01-02")
    assert df.loc[jan2, "open"] == 20.0
    assert df.loc[jan2, "close"] == 20.2

    # Confirms the loader used the first sheet, not the second.
    assert pd.Timestamp("2020-01-01") not in df.index

