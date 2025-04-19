"""定数を管理するモジュール"""

import os
from dotenv import load_dotenv

# .envファイルの読み込み
load_dotenv()

# アプリケーション設定
APP_TITLE = os.getenv("APP_TITLE", "JEPX将来価格シミュレーター")
DATA_DIR = os.getenv("DATA_DIR", os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "app_data"))

# 表示日数
SHOW_DAYS = 7

# 期間設定
MIN_YEAR = int(os.getenv("MIN_YEAR", "2024"))
MAX_YEAR = int(os.getenv("MAX_YEAR", "2045"))

# グラフ色設定
POWER_SOURCE_COLORS = {
    "原子力": "purple",
    "水力": "lightblue",
    "太陽光発電実績": "yellowgreen",
    "火力(合計)": "red",
    "風力発電実績": "green",
    "揚水": "blue",
    "蓄電池": "yellow",
    "連系線": "gray",
    "太陽光出力制御量": "yellowgreen",
    "風力出力制御量": "green",
}

# カスタムCSS
CUSTOM_CSS = """
.block-container {
    max-width: 90% !important;
    margin: auto;
    padding-top: 4rem !important;
}
h1 {
    font-size: 1.5rem !important;
    margin-bottom: 0.5rem !important;
}
.stSlider {
    padding-top: -1rem !important;
    padding-bottom: -1rem !important;
}
div[data-baseweb="slider"] {
    margin-top: -1rem !important;
    margin-bottom: -1rem !important;
}
"""

# ファイルパス設定
JEPX_DATA_PATH = os.path.join(DATA_DIR, "spot_actual_all.csv")
FUEL_DATA_PATH = os.path.join(DATA_DIR, "assumption_fuel.csv")
POWER_DATA_PATH = os.path.join(DATA_DIR, "assumption_power.csv")
TSO_DATA_TEMPLATE = os.path.join(DATA_DIR, "tso/tso_{}.csv")
