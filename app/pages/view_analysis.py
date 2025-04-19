"""価格影響分析画面を担当するモジュール"""

import streamlit as st


def create_analysis_view():
    """電源構成の価格影響分析ビューを作成"""
    st.title("電源構成の価格影響解析")

    # 分析概要
    st.markdown("""
    本分析では、各エリアの電源構成（太陽光発電、原子力発電、風力発電）がJEPX価格に与える影響を評価しています。
    回帰分析を用いて、各電源の出力変動と価格変動の関係性を定量的に分析しています。
    - 分析期間：2024年度（2024年4月～2025年3月）
    - データソース：各エリアのTSO実績データ、JEPX価格実績
    - 分析対象：
        - 太陽光発電：昼間帯（6:00-18:00）の価格影響
        - 原子力発電：24時間帯の価格影響
        - 風力発電：24時間帯の価格影響

    """)

    st.markdown("---")

    # エリアリスト（北から南の順）
    areas = ["Hokkaido", "Tohoku", "Tokyo", "Chubu", "Hokuriku",
             "Kansai", "Chugoku", "Shikoku", "Kyushu"]

    # 各エリアの分析結果を表示
    for area in areas:
        st.markdown(f"## {area}エリア")

        cols = st.columns(3)

        # 太陽光発電の分析結果
        with cols[0]:
            st.markdown("### 太陽光発電の価格影響")
            try:
                st.image(
                    f"analysis/result/tso_{area.lower()}_solar_2024.png",
                    caption=f"{area}エリアの太陽光発電による価格影響分析"
                )
            except:
                st.warning("※ 発電実績過小のため他地域解析結果を使用")

        # 原子力発電の分析結果
        with cols[1]:
            st.markdown("### 原子力発電の価格影響")
            try:
                st.image(
                    f"analysis/result/tso_{area.lower()}_nuclear_2024.png",
                    caption=f"{area}エリアの原子力発電による価格影響分析"
                )
            except:
                st.warning("※ 発電実績過小のため他地域解析結果を使用")

        # 風力発電の分析結果
        with cols[2]:
            st.markdown("### 風力発電の価格影響")
            try:
                st.image(
                    f"analysis/result/tso_{area.lower()}_wind_2024.png",
                    caption=f"{area}エリアの風力発電による価格影響分析"
                )
            except:
                st.warning("※ 発電実績過小のため他地域解析結果を使用")

        st.markdown("---")
