"""画面表示を担当するモジュール"""

import pandas as pd
from app.constants import POWER_SOURCE_COLORS, CUSTOM_CSS
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
from app.models import generate_yearly_price_data
from app.pages.view_analysis import create_analysis_view
from app.pages.view_approach import create_approach_view


def create_tab_navigation():
    """タブナビゲーションを作成"""
    return st.tabs(["JEPX長期シミュレーション", "アプローチ解説", "価格影響分析"])


def set_page_style():
    """ページのスタイルを設定"""
    st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)


def create_projection_graph(df_fuel: pd.DataFrame) -> go.Figure:
    """将来価格予測グラフを作成"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 棒グラフ: JEPX年平均
    fig.add_trace(go.Bar(
        x=df_fuel["year"],
        y=df_fuel["price_projection"],
        name="JEPX年平均(円/kWh)",
        marker_color="lightblue",
        opacity=0.7,
        text=df_fuel["price_projection"].apply(lambda x: f"{x:.1f}"),
        textposition="outside",
    ), secondary_y=True)

    # 線グラフ: 各種指標
    indices = {
        "Gas($/b),Index": {"column": "gas_dol_idx", "color": "orangered"},
        "¥/$,Index": {"column": "yen_per_dol_idx", "color": "forestgreen"},
        "Gas(¥/b),Index": {"column": "fuel_index", "color": "darkblue"}
    }

    for name, info in indices.items():
        fig.add_trace(go.Scatter(
            x=df_fuel["year"],
            y=df_fuel[info["column"]]*100,
            name=name,
            line=dict(color=info["color"], width=2)
        ), secondary_y=False)

    # レイアウト設定
    fig.update_layout(
        title="年平均推移",
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(x=0.01, y=0.1)
    )

    fig.update_yaxes(
        title_text="Index (%)",
        range=[0, 150],
        secondary_y=False
    )

    fig.update_yaxes(
        title_text="JEPX年平均(円/kWh)",
        range=[0, 20],
        secondary_y=True
    )

    return fig


def create_detail_graph(df_jepx_mst: pd.DataFrame, df_jepx: pd.DataFrame,
                        df_tso: pd.DataFrame, area: str) -> go.Figure:
    """価格・電源構成の詳細グラフを作成"""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    # 価格グラフ
    price_traces = [
        {"data": df_jepx, "column": "将来価格", "name": "将来価格 (円/kWh)"},
        {"data": df_jepx_mst, "column": area, "name": "2024価格 (円/kWh)"}
    ]

    for trace in price_traces:
        fig.add_trace(
            go.Scattergl(
                x=trace["data"]["日時"],
                y=trace["data"][trace["column"]],
                name=trace["name"],
                legendgroup="price",
                showlegend=True
            ),
            row=1, col=1,
        )

    # 電源構成グラフの表示範囲を設定
    max_power = df_tso["合計"].max()  # 2024実績の最大値を基準
    y_max = max_power * 1.5
    y_min = -max_power * 0.1  # 最大値の10%を下限に設定

    # 表示する電源種別を明示的に指定（合計は除外）
    power_sources = [
        "原子力",
        "水力",
        "太陽光発電実績",
        "火力(合計)",
        "風力発電実績",
        "揚水",
        "蓄電池",
        "連系線",
        "太陽光出力制御量",
        "風力出力制御量"
    ]

    # 正の値のトレース (積み上げ)
    positive_traces = []
    for source in power_sources:
        if source in df_tso.columns and source != "合計" and source != "太陽光出力制御量" and source != "風力出力制御量":
            # 正の値のみのマスク適用
            positive_values = df_tso[source].copy()
            positive_values[positive_values < 0] = 0

            positive_traces.append(
                go.Bar(
                    x=df_tso["datetime"],
                    y=positive_values,
                    name=source,
                    marker_color=POWER_SOURCE_COLORS[source],
                    legendgroup="power_pos",
                    showlegend=True
                )
            )

    # 正の値のトレースをまとめて追加
    for trace in positive_traces:
        fig.add_trace(trace, row=2, col=1)

    # 負の値のトレース (別途積み上げ)
    negative_traces = []
    for source in power_sources:
        if source in df_tso.columns and source != "合計":
            # 負の値のみのマスク適用
            negative_values = df_tso[source].copy()
            negative_values[negative_values > 0] = 0

            # 負の値がある場合のみ追加
            if (negative_values < 0).any():
                negative_traces.append(
                    go.Bar(
                        x=df_tso["datetime"],
                        y=negative_values,
                        name=f"{source} ",
                        marker_color=POWER_SOURCE_COLORS[source],
                        legendgroup="power_neg",
                        showlegend=True,
                        opacity=0.7
                    )
                )

    # 負の値のトレースをまとめて追加
    for trace in negative_traces:
        fig.add_trace(trace, row=2, col=1)

    # グラフ設定
    fig.update_yaxes(title_text="JEPX価格（円/kWh）", range=[0, 40], row=1, col=1)
    fig.update_yaxes(title_text="電源稼働量 (MW)", range=[
                     y_min, y_max], row=2, col=1)
    fig.update_xaxes(tickformat="%m/%d %H:%M", row=2, col=1)
    fig.update_layout(
        height=500,
        width=1300,
        margin=dict(l=20, r=20, t=50, b=20),
        barmode="relative",
        legend_tracegroupgap=50,
        legend=dict(y=0.1),
        title={
            "text": "コマ別 価格・電源稼働",
            "font": {"size": 20, "color": "black"},
        }
    )
    return fig


def create_download_button(df_jepx_2024: pd.DataFrame, df_jepx_2045: pd.DataFrame,
                           df_yearly_avg: pd.DataFrame, area: str):
    """ダウンロードボタンを作成し、クリック時に全期間データを生成してダウンロード"""
    if st.button("全期間データダウンロード"):
        with st.spinner("全期間の価格データを生成中..."):
            # 全期間のデータを生成
            df_all_years = generate_yearly_price_data(
                df_jepx_2024, df_jepx_2045, df_yearly_avg, area
            )
            # CSVに変換
            csv = df_all_years.to_csv(index=False, encoding="utf-8-sig")

        # ダウンロードボタンを表示（データ生成後に表示）
        st.download_button(
            label="生成されたデータをダウンロード",
            data=csv,
            file_name=f"jepx_price_{area}_{df_yearly_avg['year'].min()}-{df_yearly_avg['year'].max()}.csv",
            mime="text/csv",
            key='download-csv'
        )
