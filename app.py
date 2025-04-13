import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.interpolate import interp1d
import io

SHOW_DAYS = 7


def set_css():
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 90% !important;
            margin: auto;
            padding-top: 4rem !important;
        }
        h1 {
            font-size: 1.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        .stSlider, .stTable {
            margin-top: 0px !important;
            margin-bottom: 0px !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_jepx_data():
    df = pd.read_csv("./data/spot_actual_all.csv", encoding="shift_jis")
    df["日時"] = pd.to_datetime(df["受渡日"]) + pd.to_timedelta(
        (df["時刻コード"] - 1) * 30, unit="minutes"
    )
    df["時刻"] = df["日時"].dt.hour + df["日時"].dt.minute / 60
    return df


@st.cache_data
def load_tso_data(area):
    df = pd.read_csv(f"./data/tso/tso_{area}_2024.csv", encoding="shift_jis")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def get_slider_params(area, powersource):
    assumptions_df = pd.read_csv("./data/assumption_power.csv")
    filtered_df = assumptions_df[
        (assumptions_df["area"] == area) & (
            assumptions_df["powersource"] == powersource)
    ]
    if filtered_df.empty:
        return 0, 0, 10, -0.1, -0.05, -0.01
    row = filtered_df.iloc[0]
    return (
        float(row["composition_bottom"]),
        float(row["composition_base"]),
        float(row["composition_top"]),
        float(row["sensitivity_bottom"]),
        float(row["sensitivity_base"]),
        float(row["sensitivity_top"]),
    )


def update_dataframes(area, df_jepx, df_tso, nuclear_ratio, solar_ratio, fuel_idx_2045):

    nuclear_impact = get_slider_params(area, "nuclear")[4]
    solar_impact = get_slider_params(area, "solar")[4]

    tso_yearly_avg = df_tso["合計"].mean()
    df_tso["原子力"] = tso_yearly_avg * (nuclear_ratio / 100)
    bottom_solar_ratio = get_slider_params(area, 'solar')[0]

    df_jepx["将来価格"] = df_jepx[f"{area}"] + nuclear_ratio * nuclear_impact
    df_jepx["将来価格"] = df_jepx["将来価格"] + (df_tso["太陽光発電実績"] / df_tso["太陽光発電実績"].mean()) * \
        (solar_ratio - bottom_solar_ratio) * solar_impact

    df_tso["太陽光発電実績"] = df_tso["太陽光発電実績"] * \
        (solar_ratio / bottom_solar_ratio)

    # 火力比 = 火力（合計）/合計 を算出
    df_jepx["thermal_ratio"] = df_tso["火力(合計)"] / df_tso["合計"]
    df_jepx["将来価格"] = df_jepx["将来価格"] * \
        ((1 - df_jepx["thermal_ratio"]) +
         df_jepx["thermal_ratio"] * fuel_idx_2045)
    df_jepx["将来価格"] = df_jepx["将来価格"].clip(lower=0.01)

    return df_jepx, df_tso


def calculate_farmland_price(df_fuel, jepx_avg_2024, df_jepx_power_update, df_tso_power_update, oil_dol_idx_update, yen_per_dol_idx_update):

    # スライダー値に応じた当初値からの変更率を取得 e.g. 0.8
    oil_change_ratio = oil_dol_idx_update / \
        (df_fuel["gas_dol"].iloc[-1]/df_fuel["gas_dol"][0])
    # 2024年の価格を基準にして、2024年の価格を基準にした変更率を計算
    df_fuel["gas_dol_idx"] = (df_fuel["gas_dol"] /
                              df_fuel["gas_dol"][0]) * oil_change_ratio
    # 2024年だけは実績のため維持
    df_fuel["gas_dol_idx"][0] = 1.0
    # 燃料価格に適用
    df_fuel["gas_dol"] = df_fuel["gas_dol"] * df_fuel["gas_dol_idx"]

    # 為替レートの変更率を計算
    yen_per_dol_change_ratio = yen_per_dol_idx_update / \
        (df_fuel["yen_per_dol"].iloc[-1]/df_fuel["yen_per_dol"][0])
    # 2024年の価格を基準にして、2024年の価格を基準にした変更率を計算
    df_fuel["yen_per_dol_idx"] = (df_fuel["yen_per_dol"] /
                                  df_fuel["yen_per_dol"][0]) * yen_per_dol_change_ratio
    # 2024年だけは実績のため維持
    df_fuel["yen_per_dol_idx"][0] = 1.0
    # 為替レートに適用
    df_fuel["yen_per_dol"] = df_fuel["yen_per_dol"] * \
        df_fuel["yen_per_dol_idx"]

    # fuel_indexを計算
    df_fuel['fuel_index'] = df_fuel["gas_dol_idx"] * df_fuel["yen_per_dol_idx"]

    # 燃料価格分計算のために火力に反映
    ratio_thermal = df_tso_power_update["火力(合計)"].mean(
    ) / df_tso_power_update["合計"].mean()

    df_fuel["yearly_index"] = ratio_thermal * \
        df_fuel["fuel_index"] + (1 - ratio_thermal)
    df_fuel["price_projection"] = df_fuel["yearly_index"] * jepx_avg_2024

    x = [2024, 2045]
    y = [jepx_avg_2024, df_jepx_power_update["将来価格"].mean()]
    linear_interp = interp1d(x, y, kind='linear')
    normalized_values = [linear_interp(
        year) / jepx_avg_2024 for year in range(2024, 2046)]
    df_fuel["price_projection"] = df_fuel["price_projection"] * normalized_values

    return df_fuel


def plot_projection_graph(area, df_fuel, jepx_avg_2024, df_jepx_power_update, df_tso_power_update):

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    # 棒グラフ: JEPX年平均
    fig2.add_trace(go.Bar(
        x=df_fuel["year"],
        y=df_fuel["price_projection"],
        name="JEPX年平均(円/kWh)",
        marker_color="lightblue",
        opacity=0.6,
        text=df_fuel["price_projection"].apply(
            lambda x: f"{x:.1f}"),  # 少数1桁にフォーマット
        textposition="outside"  # 数値を棒の上に表示
    ), secondary_y=True)

    # 線グラフ1: Gas Index
    fig2.add_trace(go.Scatter(
        x=df_fuel["year"],
        y=df_fuel["gas_dol_idx"]*100,
        name="Gas(USD/b,Index(%))"
    ), secondary_y=False)

    # 線グラフ2: Yen per Dollar Index
    fig2.add_trace(go.Scatter(
        x=df_fuel["year"],
        y=df_fuel["yen_per_dol_idx"]*100,
        name="¥/$(Index(%))"
    ), secondary_y=False)

    # 線グラフ3: Gas Index
    fig2.add_trace(go.Scatter(
        x=df_fuel["year"],
        y=df_fuel["fuel_index"]*100,
        name="Fuel Index(¥based,%)"
    ), secondary_y=False)

    # レイアウト設定
    fig2.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(x=0.01, y=0.1)
    )

    # Y軸のタイトル設定
    fig2.update_yaxes(title_text="Index (%)",
                      range=[0, 150],
                      secondary_y=False)  # 左側のY軸

    # グラフを描画
    st.plotly_chart(fig2, use_container_width=True)
    return df_fuel["price_projection"].iloc[-1]


def plot_graphs(df_jepx_mst, df_jepx, df_tso, df_fuel, area, start_date, end_date):

    df_tso_show = df_tso[
        (df_tso["datetime"] >= pd.Timestamp(start_date))
        & (df_tso["datetime"] <= pd.Timestamp(end_date))
    ]

    df_jepx_mst_show = df_jepx_mst[
        (df_jepx_mst["日時"] >= pd.Timestamp(start_date)) & (
            df_jepx_mst["日時"] <= pd.Timestamp(end_date))
    ]

    # 燃料価格分計算のために火力に反映
    ratio_thermal = df_tso_show["火力(合計)"].mean() / df_tso_show["合計"].mean()

    df_jepx["将来価格"] = df_jepx["将来価格"] * \
        (ratio_thermal * df_fuel["fuel_index"].iloc[-1] + (1 - ratio_thermal))

    df_jepx_show = df_jepx[
        (df_jepx["日時"] >= pd.Timestamp(start_date)) & (
            df_jepx["日時"] <= pd.Timestamp(end_date))
    ]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        # subplot_titles=("コマ別価格", "コマ別電源稼働"),
        vertical_spacing=0.1,
    )

    fig.add_trace(
        go.Scattergl(x=df_jepx_show["日時"],
                     y=df_jepx_show["将来価格"], name="将来価格 (円/kWh)",
                     legendgroup="price",
                     showlegend=True
                     ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scattergl(x=df_jepx_mst_show["日時"],
                     y=df_jepx_mst_show[f"{area}"], name="2024価格 (円/kWh)",
                     legendgroup="price",
                     showlegend=True
                     ),
        row=1, col=1,
    )

    fig.update_yaxes(
        title_text="JEPX価格（円/kWh）",
        range=[0, 50],
        row=1, col=1
    )
    fig.update_xaxes(
        tickformat="%m/%d %H:%M",
        row=2, col=1
    )

    colors = {
        "原子力": "purple", "水力": "lightblue", "太陽光発電実績": "yellowgreen", "火力(合計)": "red",
        "風力発電実績": "green", "揚水": "blue", "蓄電池": "yellow", "連系線": "gray",
        "太陽光出力制御量": "brown", "風力出力制御量": "brown",
    }

    max120 = df_tso_show["合計"].max() * 1.2

    for col in colors.keys():
        if col in df_tso_show.columns:
            fig.add_trace(
                go.Bar(x=df_tso_show["datetime"], y=df_tso_show[col],
                       name=col, marker_color=colors[col],
                       legendgroup="power",
                       showlegend=True  #
                       ),
                row=2,
                col=1,
            )

    # コマ別電源稼働のY軸設定
    fig.update_yaxes(
        title_text="電源稼働量 (MW)",  # Y軸のタイトル
        range=[0, max120],  # Y軸の範囲を設定
        row=2,  # 2行目のグラフ
        col=1   # 対象の列番号
    )

    fig.update_layout(
        height=600, width=1200, margin=dict(
            l=20, r=20, t=50, b=20),
        barmode="stack",
        legend_tracegroupgap=100,
        legend=dict(y=0.1),
        title={
            "text": "コマ別 価格・電源稼働",
            "font": {"size": 20, "color": "black"}, }

    )

    # fig.update_layout(width=800)  # グラフの幅を固定
    st.plotly_chart(fig, use_container_width=False)  # 自動幅調整を無効化

    # st.plotly_chart(fig, use_container_width=True)
    return df_jepx


def main():
    set_css()
    df_jepx_mst = load_jepx_data()
    min_date, max_date = df_jepx_mst["日時"].min(), df_jepx_mst["日時"].max()
    start_date = min_date
    end_date = start_date + pd.Timedelta(days=SHOW_DAYS)

    st.header("JEPX将来価格検証")

    header_b, header_c, header_c2, header_d, header_e, header_f = st.columns([
        2, 2, 1, 1, 1, 1])

    with header_b:
        area = st.selectbox(
            "エリアを選択してください(現在Tokyoのみ)",
            [
                # "Hokkaido",
                # "Tohoku",
                "Tokyo",
                # "Chubu",
                # "Hokuriku",
                # "Kansai",
                # "Chugoku",
                # "Shikoku",
                # "Kyushu"
            ],
            index=0,
        )
    with header_c:
        start_date = st.date_input(
            "コマ別表示開始日を選択してください（2024基準）", min_date, min_value=min_date, max_value=max_date
        )
        end_date = start_date + pd.Timedelta(days=SHOW_DAYS)

    col_left, col_right = st.columns([1, 1])

    jepx_avg_2024 = df_jepx_mst[area].mean()
    df_tso_mst = load_tso_data(area)
    df_fuel_mst = pd.read_csv("./data/assumption_fuel.csv")
    fuel_index_2045_mst = df_fuel_mst["fuel_index"].iloc[-1]

    with col_left:
        st.subheader("前提シナリオ")
        col_power, col_fuel = st.columns([1, 1])
        with col_power:
            nuclear_params = get_slider_params(area, "nuclear")
            solar_params = get_slider_params(area, "solar")
            st.markdown("<h5>電源構成の主変化（全体容量比）</h5>",
                        unsafe_allow_html=True)
            # 原子力比率スライダー
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between;">
                    <span>2024現状</span>
                    <span>〜</span>
                    <span>系統計画(2040頃)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")

            # 原子力比率スライダー
            nuclear_ratio = st.slider(
                "原子力(%)", nuclear_params[0], nuclear_params[2], nuclear_params[1]
            )
            # 太陽光比率スライダー
            solar_ratio = st.slider(
                "太陽光(%)", solar_params[0], solar_params[2], solar_params[1]
            )
        with col_fuel:
            st.markdown("<h5>燃料価格・為替の変化</h5>",
                        unsafe_allow_html=True)

            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between;">
                    <span>2024現状</span>
                    <span>〜</span>
                    <span>2045推計</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("")
            gas_dol_2045 = float(
                df_fuel_mst["gas_dol"].iloc[-1]/df_fuel_mst["gas_dol"][0])
            yen_per_dol_2045 = float(
                df_fuel_mst["yen_per_dol"].iloc[-1]/df_fuel_mst["yen_per_dol"][0])

            # 燃料価格スライダー
            oil_dol_idx_update = st.slider(
                "GAS(USD/b)　24年比", 1.0, gas_dol_2045, gas_dol_2045
            )
            # 為替スライダー
            yen_per_dol_idx_update = st.slider(
                "為替(¥/$)　24年比", 0.5, yen_per_dol_2045, yen_per_dol_2045
            )
    # スライダーの値とプロット用の更新データフレームを取得
    df_jepx_power_update, df_tso_power_update = update_dataframes(area, df_jepx_mst, df_tso_mst,
                                                                  nuclear_ratio, solar_ratio, fuel_index_2045_mst)
    df_fuel_update = calculate_farmland_price(
        df_fuel_mst, jepx_avg_2024, df_jepx_power_update, df_tso_power_update,
        oil_dol_idx_update, yen_per_dol_idx_update)

    with col_right:
        st.subheader("年平均推計")
        st.markdown("<h5>エリアスポット取引価格 2024-2045</h5>",
                    unsafe_allow_html=True)
        price_projection_2045 = plot_projection_graph(
            area, df_fuel_update, jepx_avg_2024, df_jepx_power_update, df_tso_power_update)

    df_jepx_result = plot_graphs(df_jepx_mst, df_jepx_power_update, df_tso_power_update, df_fuel_update,
                                 area, start_date, end_date)

    with header_d:
        st.metric(label="2024平均価格 (円/kWh)", value=round(jepx_avg_2024, 2))
    with header_e:
        st.metric(label="2045想定価格 (円/kWh)",
                  value=round(price_projection_2045, 2))
    with header_f:
        # データフレームをCSV形式に変換
        csv = df_jepx_result.to_csv(index=False, encoding="utf-8-sig")

        # ダウンロードボタンを追加
        st.download_button(
            label="ダウンロード (未実装)",  # ボタンのラベル
            data=csv,  # ダウンロードするデータ
            file_name="jepx_est_result.csv",  # ダウンロードされるファイル名
            mime="text/csv",  # ファイル形式
        )

    st.markdown(
        """
    <div style="margin-top: 40px;"></div>  <!-- 上にスペースを追加 -->
    補足/確認事項
    <ul style="line-height: 1.2;color: gray;">
        電源
        <li>計画はOCCTOマスタープラン2023（エリア別最新）</li>
        <li>原子力はベース、太陽光は実発電量に基づき変動。他優先考慮事項確認</li>
        <li>電源増減∝価格は過去実績元に算出予定（可視化含む）</li>
        燃料
        <li> LNGに他燃料も連動。為替を考慮し火力稼働分に適用</li>
        <li>【確認】LNG：XXX公表推計。為替：XXX公表推計</li>
        <li>【確認】どういうシナリオとするか >> 横軸変更</li>
        コマ別

    </ul>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
    st.experimental_set_query_params(scroll_to_top="true")
