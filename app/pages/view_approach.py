"""アプローチ解説画面を担当するモジュール"""

import streamlit as st


def create_approach_view():
    """アプローチ解説ビューを作成"""
    st.title("アプローチ解説")
    st.markdown("""
    ### サマリー
    - **電源構成の影響評価**: 原子力・太陽光の発電量変化による価格変動を、エリア別に定量的に評価
    - **需給運用のモデル化**: 余剰発電を他エリアへの連系線活用、揚水発電での夜間活用など実運用を考慮
    - **複数シナリオでの分析**: 燃料価格・為替・電源構成を組み合わせた2045年までの価格推移検証

    ### 計算の仕組み

    #### 想定事象１：電源構成の変化
    - **電源構成**: 原子力・太陽光・風力が全体に占める割合を第七次エネ基を上限として任意に設定
        - 第７次エネ基（全国）の各電源増加率を各エリアの現状に適用（https://www.enecho.meti.go.jp/category/others/basic_plan/pdf/20250218_02.pdf　p.9）
        - 但し、原子力は再稼働見通し等元に2045年時上限を作成（https://www.enecho.meti.go.jp/category/electricity_and_gas/nuclear/001/　）
    - **余剰電力の活用**: 原子力・太陽光・風力の増分は、連系線（24実績上限まで）→ 火力抑制（24実績下限まで） → 揚水（24実績最大容量まで） → 再エネ出力制御　の順で調整
    - **価格影響係数**: 原子力・太陽光・風力の増減による価格影響は、エリアごとの回帰分析から算出（”価格影響分析”参照）

    #### 想定事象２：燃料・為替の変動
    - **影響範囲**: 将来の燃料価格と為替の変化が、火力稼働分に影響を与えることを想定（価格影響は火力が占める割合のみ）
    - **燃料価格**
        - World Energy Outlook 2024(Japan） https://www.iea.org/reports/world-energy-outlook-2024
        - Think Tank：Rystad Energy社公表値　 https://www.rystadenergy.com/insights/whitepaper-forecasting-future-gas-prices
        - U.S. Energy Information Administration Annuarl Energy Outlook 2025 (HenryHub, 単一シナリオ) https://www.eia.gov/outlooks/aeo/?utm_source=chatgpt.com
        - 独自の長期LNG価格想定
    - **為替**: 2024年平均151円に対する比率として設定。最長期でも2035年までの想定であるため、35年まで線形変化ののち横置き
        - Base: 106円 (24年比0.7)：MUFG「日本経済の中期見通し」https://www.murc.jp/wp-content/uploads/2023/09/medium_2309_01.pdf
        - High: 165円 (24年比1.1)：第一生命経済研究所「AIが予測するトランプ政権下のドル円相場」基本シナリオ https://www.dlri.co.jp/report/ld/387997.html?utm_source=chatgpt.com
        - Low: 90円 （24年比0.6）：同上、円高シナリオ
        

    ### その他留意事項
    - **需要想定**: 本手法では考慮の必要がないものとする
        - 過去データ解析及び価格変動の考慮は全て、全体に対する電源構成の”比率”で実施している。
        - エネルギー基本計画も構成比で示されている。従い需要量・供給の絶対量変化は考慮する必要がない。
    - **カーボンプライス**: 現状未反映だがシナリオ考慮可能（2050年想定 0.8~2.5円/kWh 詳細略）
    - **蓮系線**: 理想系としての考慮は可能（系統計画に基づき容量を拡大。但し他地域連動考慮は不可のため、対象エリア向けの理想挙動とするor一定制約をおいて反映可能）
    """)
