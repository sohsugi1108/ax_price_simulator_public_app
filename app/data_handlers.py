"""データ処理を担当するモジュール"""

import pandas as pd
import streamlit as st
from app.constants import JEPX_DATA_PATH, TSO_DATA_TEMPLATE, FUEL_DATA_PATH, POWER_DATA_PATH


@st.cache_data
def load_jepx_data() -> pd.DataFrame:
    """JEPXデータを読み込み、前処理を行う"""
    df = pd.read_csv(JEPX_DATA_PATH, encoding="shift_jis")
    df["日時"] = pd.to_datetime(df["受渡日"]) + pd.to_timedelta(
        (df["時刻コード"] - 1) * 30, unit="minutes"
    )
    df["時刻"] = df["日時"].dt.hour + df["日時"].dt.minute / 60

    # 2024年度のデータのみを抽出
    df = df[(df["日時"] >= "2024-04-01") & (df["日時"] < "2025-04-01")]

    return df


@st.cache_data
def load_tso_data(area: str) -> pd.DataFrame:
    """送電事業者データを読み込む"""
    df = pd.read_csv(TSO_DATA_TEMPLATE.format(area), encoding="utf-8")
    df["datetime"] = pd.to_datetime(df["datetime"])

    # 2020年のデータを2024年に変換（4年シフト）
    df["datetime"] = df["datetime"] + pd.DateOffset(years=4)

    # 2024年度のデータのみを抽出（4/1-3/31）
    df = df[(df["datetime"] >= "2024-04-01") & (df["datetime"] < "2025-04-01")]

    return df


@st.cache_data
def load_fuel_data() -> pd.DataFrame:
    """燃料データを読み込む"""
    return pd.read_csv(FUEL_DATA_PATH)


@st.cache_data
def get_power_source_parameters(area: str, power_source: str) -> tuple:
    """電源パラメータを取得する"""
    assumptions_df = pd.read_csv(POWER_DATA_PATH)
    filtered_df = assumptions_df[
        (assumptions_df["area"] == area) &
        (assumptions_df["powersource"] == power_source)
    ]

    if filtered_df.empty:
        return 0, 0, 10, -0.1, -0.05, -0.01

    row = filtered_df.iloc[0]
    return (
        float(row["composition_bottom"]),
        float(row["composition_base"]),
        float(row["composition_top"]),
        float(row["impact"]),
    )


@st.cache_data
def filter_data_by_date_range(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp,
                              datetime_column: str = "日時") -> pd.DataFrame:
    """指定された日付範囲でデータをフィルタリング"""
    return df[
        (df[datetime_column] >= pd.Timestamp(start_date)) &
        (df[datetime_column] <= pd.Timestamp(end_date))
    ]
