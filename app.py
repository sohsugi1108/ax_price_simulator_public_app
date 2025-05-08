"""JEPX将来価格シミュレーターのメインモジュール"""

import streamlit as st
import pandas as pd
from app.constants import SHOW_DAYS
from app.data_handlers import (
    load_jepx_data,
    load_tso_data,
    load_fuel_data,
    get_power_source_parameters,
    filter_data_by_date_range
)
from app.models import update_power_composition, calculate_fuel_price_projection
from app.views import (
    set_page_style,
    create_projection_graph,
    create_detail_graph,
    create_download_button,
    create_tab_navigation,
    create_analysis_view,
    create_approach_view  # 追加
)


def main():
    """メイン処理"""
    # ページスタイルとタブの設定
    set_page_style()
    sim_tab, approach_tab, analysis_tab = create_tab_navigation()  # approach_tab を追加

    # 分析タブの表示
    with analysis_tab:
        create_analysis_view()

    # アプローチ解説タブの表示
    with approach_tab:
        create_approach_view()  # 追加

    # シミュレーションタブの表示（メイン機能）
    with sim_tab:
        # 基本データの読み込み
        df_jepx_mst = load_jepx_data()
        min_date, max_date = df_jepx_mst["日時"].min(), df_jepx_mst["日時"].max()
        start_date = min_date
        end_date = start_date + pd.Timedelta(days=SHOW_DAYS)

        # エリア選択と日付入力のレイアウト
        header_b, header_c, header_c2, header_d, header_e, header_f = st.columns([
            2, 2, 1, 1, 1, 1])

        with header_b:
            area = st.selectbox(
                "エリア選択(一部エリア系統〆待ち)",
                ["Tokyo", "Hokkaido", "Chubu",
                    "Kansai", "Chugoku", "Kyushu",
                    #  "Shikoku",
                    # "Tohoku",
                    # "Hokuriku",
                 ],
                index=0,
            )
        with header_c:
            start_date = st.date_input(
                "コマ別表示の開始日選択（2024基準）",
                min_date,
                min_value=min_date,
                max_value=max_date
            )
            end_date = start_date + pd.Timedelta(days=SHOW_DAYS)

        # メインコンテンツのレイアウト
        col_left, col_right = st.columns([1, 1])

        # 基準データの準備
        jepx_avg_2024 = df_jepx_mst[area].mean()
        df_tso_mst = load_tso_data(area)
        df_fuel_mst = load_fuel_data()
        fuel_index_2045_mst = df_fuel_mst["fuel_index"].iloc[-1]

        # 前提シナリオの入力UI
        with col_left:
            # st.markdown("<h5>前提：2045年シナリオ想定</h5>", unsafe_allow_html=True)
            col_power, col_fuel = st.columns([1, 1])

            # 電源構成の設定
            with col_power:

                nuclear_params = get_power_source_parameters(area, "nuclear")
                solar_params = get_power_source_parameters(area, "solar")
                wind_params = get_power_source_parameters(area, "wind")

                st.markdown("<b>2045 電源比率（kWhベース,%）</b>",
                            unsafe_allow_html=True)
                st.markdown(
                    """
                    <div style="display: flex; justify-content: space-between;">
                        <span>24年水準</span>
                        <span>中庸</span>
                        <span>エネ基</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                nuclear_ratio = st.slider(
                    "原子力",
                    nuclear_params[0],
                    nuclear_params[2],
                    nuclear_params[1]
                )

                solar_ratio = st.slider(
                    "太陽光",
                    solar_params[0],
                    solar_params[2],
                    solar_params[1]
                )

                wind_ratio = st.slider(
                    "風力",
                    wind_params[0],
                    wind_params[2],
                    wind_params[1]
                )

            # 燃料価格・為替の設定
            with col_fuel:
                st.markdown("<b>2045 為替・燃料</b>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div style="display: flex; justify-content: space-between;">
                        <span>Low(90円)</span>
                        <span>MUFG推計(105円)</span>
                        <span>High(165円)</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                yen_per_dol_idx_update = st.slider(
                    "為替：¥/$,24年比",
                    0.6,
                    1.1,
                    0.7,
                )
                # GAS価格シナリオの選択
                scenario_options = {
                    "World Energy Outlook 2024: Steps(Japan)": "gas_dol_weo_steps",
                    "Think Tank(EastAsia)": "gas_dol_rystad_EastAsia",
                    "Think Tank(TTF)": "gas_dol_rystad_TTF",
                    "U.S. Annual Energy Outlook 2025:(HenryHub)": "gas_dol_aeo",
                    "Original Internal Data(FYI)": "gas_dol"
                }
                selected_scenario = st.radio(
                    "燃料価格（LNG,$/b）",
                    list(scenario_options.keys()),
                    horizontal=True,  # 横並びで表示
                    index=2
                )
                selected_column = scenario_options[selected_scenario]

        # データ更新処理
        # 電源構成の影響を考慮して燃料価格の将来推計を計算
        df_fuel_update = calculate_fuel_price_projection(
            df_fuel_mst, jepx_avg_2024, df_jepx_mst, df_tso_mst,
            selected_column, yen_per_dol_idx_update,
            nuclear_ratio, solar_ratio, wind_ratio, area
        )

        # コマ別の価格と電源構成を更新
        df_jepx_power_update, df_tso_power_update = update_power_composition(
            area, df_jepx_mst, df_tso_mst,
            nuclear_ratio, solar_ratio, wind_ratio, df_fuel_update["fuel_index"].iloc[-1]
        )

        # 年平均推計グラフの表示
        with col_right:
            fig_projection = create_projection_graph(df_fuel_update)
            st.plotly_chart(fig_projection, use_container_width=True)
            price_projection_2045 = df_fuel_update["price_projection"].iloc[-1]

        # 期間指定データの準備
        df_tso_filtered = filter_data_by_date_range(
            df_tso_power_update, start_date, end_date, "datetime"
        )
        df_jepx_filtered = filter_data_by_date_range(
            df_jepx_power_update, start_date, end_date
        )
        df_jepx_mst_filtered = filter_data_by_date_range(
            df_jepx_mst, start_date, end_date
        )

        # 詳細グラフの表示
        fig_detail = create_detail_graph(
            df_jepx_mst_filtered, df_jepx_filtered,
            df_tso_filtered, area
        )
        st.plotly_chart(fig_detail, use_container_width=False)

        # 指標値の表示
        with header_d:
            st.metric(label="2024平均価格 (円/kWh)", value=round(jepx_avg_2024, 2))
        with header_e:
            st.metric(label="2045想定価格 (円/kWh)",
                      value=round(price_projection_2045, 2))
        with header_f:
            # ダウンロードボタンに渡す引数を変更
            create_download_button(
                df_jepx_mst, df_jepx_power_update, df_fuel_update, area
            )
        st.markdown(
            """
            <div style='font-size: 1.0em; text-align: right; font-style: italic;'>
                    Powered by 
                    <span style='color: #666666; font-weight: 500;'>JER</span><span style='color: green; font-weight: 500;'>ACross</span><span style='color: #007BA7; font-weight: 500;'>Digital</span>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
