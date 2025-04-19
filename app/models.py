"""ビジネスロジックを担当するモジュール"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import interp1d
from app.data_handlers import get_power_source_parameters


@st.cache_data
def update_power_composition(area: str, df_jepx_input: pd.DataFrame, df_tso_input: pd.DataFrame,
                             nuclear_ratio: float, solar_ratio: float, wind_ratio: float,
                             fuel_idx_2045: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """電源構成を更新し、価格への影響を計算"""
    df_jepx = df_jepx_input.copy()
    df_tso = df_tso_input.copy()

    # パラメータの取得（一度に取得して再利用）
    nuclear_params = get_power_source_parameters(area, "nuclear")
    solar_params = get_power_source_parameters(area, "solar")
    wind_params = get_power_source_parameters(area, "wind")
    nuclear_impact, bottom_nuclear_ratio = nuclear_params[3], nuclear_params[0]
    solar_impact, bottom_solar_ratio = solar_params[3], solar_params[0]
    wind_impact, bottom_wind_ratio = wind_params[3], wind_params[0]

    # 電源構成の更新（ベクトル化された操作）
    tso_yearly_avg = df_tso["合計"].mean()
    original_nuclear = df_tso["原子力"].copy()
    original_solar = df_tso["太陽光発電実績"].copy()
    original_wind = df_tso["風力発電実績"].copy()
    original_pumped = df_tso["揚水"].copy()

    # パラメータの安全な値に修正
    safe_bottom_nuclear = np.maximum(bottom_nuclear_ratio, 0.1)  # 最小値0.1%を保証
    safe_nuclear_ratio = np.maximum(nuclear_ratio, 0.1)  # 入力値も0.1%以上を保証
    safe_bottom_solar = np.maximum(bottom_solar_ratio, 0.1)
    safe_solar_ratio = np.maximum(solar_ratio, 0.1)
    safe_bottom_wind = np.maximum(bottom_wind_ratio, 0.1)
    safe_wind_ratio = np.maximum(wind_ratio, 0.1)

    # 原子力の更新（安全な計算処理）
    nuclear_mask = safe_bottom_nuclear > 5.0
    nuclear_scale = safe_nuclear_ratio / safe_bottom_nuclear

    # 原子力出力の更新
    df_tso["原子力"] = np.where(
        nuclear_mask,
        df_tso["原子力"] * nuclear_scale,
        tso_yearly_avg * (safe_nuclear_ratio / 100)
    )

    # 太陽光の更新（安全な計算処理）
    solar_scale = safe_solar_ratio / safe_bottom_solar
    df_tso["太陽光発電実績"] *= solar_scale

    # 風力の更新（原子力と同様のロジック）
    wind_mask = safe_bottom_wind > 5.0
    wind_scale = safe_wind_ratio / safe_bottom_wind

    # 風力出力の更新
    df_tso["風力発電実績"] = np.where(
        wind_mask,
        df_tso["風力発電実績"] * wind_scale,
        tso_yearly_avg * (safe_wind_ratio / 100)
    )

    # データの整合性チェック
    df_tso["原子力"] = df_tso["原子力"].clip(lower=0)  # 負の値を防止
    df_tso["太陽光発電実績"] = df_tso["太陽光発電実績"].clip(lower=0)
    df_tso["風力発電実績"] = df_tso["風力発電実績"].clip(lower=0)

    # 下限値の設定
    limits = {
        'thermal': df_tso["火力(合計)"].min(),
        'interconnection': df_tso["連系線"].min(),
        'pumped': df_tso["揚水"].min()
    }

    # 揚水の日合計下限値を計算（ベクトル化）
    df_tso['date'] = df_tso["datetime"].dt.date
    daily_negative = df_tso.groupby('date').agg({
        '揚水': lambda x: x[x < 0].sum()
    })
    daily_negative_limit = daily_negative['揚水'].min()

    # 発電増加分の計算
    total_increase = (df_tso["太陽光発電実績"] + df_tso["原子力"] + df_tso["風力発電実績"]) - \
        (original_solar + original_nuclear + original_wind)

    # 一括処理による最適化
    mask = total_increase > 0
    if mask.any():
        # 連系線での相殺（ベクトル化）
        available_interconnection = df_tso["連系線"].values - \
            limits['interconnection']
        interconnection_offset = np.minimum(
            total_increase, available_interconnection)
        df_tso.loc[mask, "連系線"] -= interconnection_offset[mask]
        total_increase[mask] -= interconnection_offset[mask]

        # 火力での相殺（ベクトル化）
        still_surplus = total_increase > 0
        if still_surplus.any():
            available_thermal = df_tso["火力(合計)"].values - limits['thermal']
            thermal_offset = np.minimum(total_increase, available_thermal)
            df_tso.loc[still_surplus,
                       "火力(合計)"] -= thermal_offset[still_surplus]
            total_increase[still_surplus] -= thermal_offset[still_surplus]

            # 揚水での相殺（ベクトル化）
            still_surplus = total_increase > 0
            if still_surplus.any():
                # 日付ごとの揚水制約を計算
                df_tso['date'] = df_tso["datetime"].dt.date
                daily_pumped = df_tso.groupby('date')['揚水'].transform(
                    lambda x: x[x < 0].sum())
                available_pumped = np.minimum(
                    df_tso["揚水"].values - limits['pumped'],
                    -(daily_negative_limit - daily_pumped)
                )
                pumped_offset = np.minimum(total_increase, available_pumped)
                df_tso.loc[still_surplus, "揚水"] -= pumped_offset[still_surplus]
                total_increase[still_surplus] -= pumped_offset[still_surplus]

                # 残余の出力制御（ベクトル化）
                still_surplus = total_increase > 0
                if still_surplus.any():
                    control_solar = df_tso.loc[still_surplus,
                                               "太陽光発電実績"] / (df_tso.loc[still_surplus, "太陽光発電実績"] + df_tso.loc[still_surplus, "風力発電実績"])
                    control_wind = 1 - control_solar

                    df_tso.loc[still_surplus,
                               "太陽光出力制御量"] -= total_increase[still_surplus] * control_solar
                    df_tso.loc[still_surplus,
                               "風力出力制御量"] -= total_increase[still_surplus] * control_wind

    # 揚水発電の増加処理（ベクトル化）
    pumped_decrease = (df_tso["揚水"] - original_pumped).clip(upper=0)
    evening_mask = df_tso["datetime"].dt.hour >= 17

    # 日付ごとに揚水発電を調整
    for date in df_tso['date'].unique():
        date_mask = (df_tso['date'] == date) & evening_mask
        date_decrease = pumped_decrease[df_tso['date'] == date].sum()

        if date_decrease < 0 and date_mask.any():
            evening_count = date_mask.sum()
            increase_per_hour = -date_decrease * 0.75 / evening_count

            # 発電増加の適用
            df_tso.loc[date_mask, "揚水"] += increase_per_hour

            # 連系線と火力での相殺（ベクトル化）
            available_interconnection = df_tso.loc[date_mask, "連系線"].values - \
                limits['interconnection']
            interconnection_offset = np.minimum(
                increase_per_hour, available_interconnection)
            df_tso.loc[date_mask, "連系線"] -= interconnection_offset

            # 残りを火力で相殺
            remaining = increase_per_hour - interconnection_offset
            available_thermal = df_tso.loc[date_mask,
                                           "火力(合計)"].values - limits['thermal']
            thermal_offset = np.minimum(remaining, available_thermal)
            df_tso.loc[date_mask, "火力(合計)"] -= thermal_offset

    # 価格影響の計算
    base_price = df_jepx[f"{area}"].values
    future_price = base_price.copy()

    # 原子力の影響
    nuclear_ratio_change = (
        nuclear_ratio - bottom_nuclear_ratio) * nuclear_impact
    nuclear_positive_mean = df_tso["原子力"][df_tso["原子力"] > 0].mean()
    nuclear_ratio_effect = np.where(nuclear_ratio == 0, 1.0,
                                    np.where(nuclear_positive_mean == 0, 1.0,
                                             df_tso["原子力"].values / nuclear_positive_mean))
    future_price += nuclear_ratio_effect * nuclear_ratio_change

    # 太陽光の影響
    solar_mean = df_tso["太陽光発電実績"].mean()
    solar_mean = solar_mean if solar_mean != 0 else 1
    solar_ratio_effect = df_tso["太陽光発電実績"].values / solar_mean
    solar_effect = (solar_ratio - bottom_solar_ratio) * solar_impact
    future_price += solar_ratio_effect * solar_effect

    # 風力の影響（原子力と同様のロジック）
    wind_ratio_change = (wind_ratio - bottom_wind_ratio) * wind_impact
    wind_positive_mean = df_tso["風力発電実績"][df_tso["風力発電実績"] > 0].mean()
    wind_ratio_effect = np.where(wind_ratio == 0, 1.0,
                                 np.where(wind_positive_mean == 0, 1.0,
                                          df_tso["風力発電実績"].values / wind_positive_mean))
    future_price += wind_ratio_effect * wind_ratio_change

    # 火力の影響
    thermal_ratio = (df_tso["火力(合計)"] / df_tso["合計"]).values
    future_price *= (1 - thermal_ratio) + (thermal_ratio * fuel_idx_2045)

    # 計算結果をDataFrameに反映
    df_jepx["将来価格"] = future_price
    df_jepx["将来価格"] = df_jepx["将来価格"].clip(lower=0.01)

    return df_jepx, df_tso


@st.cache_data
def calculate_fuel_price_projection(df_fuel_input: pd.DataFrame, jepx_avg_2024: float,
                                    df_jepx: pd.DataFrame, df_tso: pd.DataFrame,
                                    price_scenario_name: str, exchange_rate_ratio: float,
                                    nuclear_ratio: float = None, solar_ratio: float = None,
                                    wind_ratio: float = None, area: str = None) -> pd.DataFrame:
    """燃料価格の将来推計を計算（ベクトル化処理）"""
    df_fuel = df_fuel_input.copy()

    # ガス価格インデックスの計算（ベクトル化）
    base_price = df_fuel[price_scenario_name].iloc[0]
    df_fuel["gas_dol_idx"] = df_fuel[price_scenario_name] / base_price
    df_fuel.iloc[0, df_fuel.columns.get_loc("gas_dol_idx")] = 1.0

    # 為替レートの計算（ベクトル化）
    years = df_fuel["year"].values
    yearly_change = (exchange_rate_ratio - 1.0) / (2035 - 2024)

    # 条件に基づいて為替レートインデックスを計算
    conditions = [
        (years < 2024),
        (years >= 2024) & (years < 2036),
        (years >= 2036)
    ]
    choices = [
        1.0,
        1.0 + yearly_change * (years - 2024),
        exchange_rate_ratio
    ]
    df_fuel["yen_per_dol_idx"] = np.select(conditions, choices, default=1.0)

    # インデックスの計算（ベクトル化）
    df_fuel["yen_per_dol"] *= df_fuel["yen_per_dol_idx"]
    df_fuel["fuel_index"] = df_fuel["gas_dol_idx"] * df_fuel["yen_per_dol_idx"]

    # 価格推計の基本値計算（ベクトル化）
    thermal_ratio = df_tso["火力(合計)"].mean() / df_tso["合計"].mean()
    df_fuel["yearly_index"] = thermal_ratio * \
        df_fuel["fuel_index"] + (1 - thermal_ratio)
    df_fuel["price_projection"] = jepx_avg_2024 * df_fuel["yearly_index"]

    # 電源構成の影響を考慮（ベクトル化）
    if all([param is not None for param in [nuclear_ratio, solar_ratio, wind_ratio, area]]):
        nuclear_params = get_power_source_parameters(area, "nuclear")
        solar_params = get_power_source_parameters(area, "solar")
        wind_params = get_power_source_parameters(area, "wind")

        # 効果の計算
        nuclear_effect = nuclear_ratio * nuclear_params[3]
        solar_effect = (solar_ratio - solar_params[0]) * solar_params[3]
        wind_effect = (wind_ratio - wind_params[0]) * wind_params[3]

        # 進捗率をベクトル化
        progress = np.where(years < 2024, 0,
                            np.minimum(1, (years - 2024) / (2045 - 2024)))

        # 効果を一括で適用
        total_effect = (nuclear_effect + solar_effect + wind_effect) * progress
        df_fuel["price_projection"] += total_effect

    return df_fuel


def generate_yearly_price_data(df_jepx_2024: pd.DataFrame, df_jepx_2045: pd.DataFrame,
                               df_yearly_avg: pd.DataFrame, area: str) -> pd.DataFrame:
    """2024年から2045年までの各年の価格データを生成"""
    all_years_data = []
    start_year = 2024
    end_year = 2045

    # 2024年と2045年の価格データを準備
    price_2024 = df_jepx_2024[area].values
    price_2045 = df_jepx_2045["将来価格"].values

    # 年平均価格を辞書に変換
    avg_prices = df_yearly_avg.set_index('year')['price_projection'].to_dict()

    # 2024年のデータを追加
    df_2024_output = df_jepx_2024[['日時']].copy()
    df_2024_output['エリア'] = area
    df_2024_output['価格'] = price_2024
    all_years_data.append(df_2024_output)

    # 2025年から2044年までのデータを生成
    for year in range(start_year + 1, end_year):
        # 線形補間
        ratio = (year - start_year) / (end_year - start_year)
        interpolated_price = price_2024 + (price_2045 - price_2024) * ratio

        # 年平均価格で補正
        current_avg = np.mean(interpolated_price)
        target_avg = avg_prices.get(year, current_avg)  # 年平均データがない場合は補間値の平均を使う

        if current_avg > 0:  # ゼロ除算を回避
            correction_factor = target_avg / current_avg
            corrected_price = interpolated_price * correction_factor
        else:
            corrected_price = interpolated_price  # 平均が0の場合は補正しない

        # 0.01円未満をクリップ
        corrected_price = np.maximum(corrected_price, 0.01)

        # DataFrameを作成
        df_year = df_jepx_2024[['日時']].copy()
        # 日時を該当年に更新 (月日は2024年のものを流用)
        df_year['日時'] = df_year['日時'].apply(lambda dt: dt.replace(year=year))
        df_year['エリア'] = area
        df_year['価格'] = corrected_price
        all_years_data.append(df_year)

    # 2045年のデータを追加
    df_2045_output = df_jepx_2045[['日時']].copy()
    # 日時を2045年に更新
    df_2045_output['日時'] = df_2045_output['日時'].apply(
        lambda dt: dt.replace(year=end_year))
    df_2045_output['エリア'] = area
    df_2045_output['価格'] = price_2045
    df_2045_output['価格'] = df_2045_output['価格'].clip(lower=0.01)  # 2045年もクリップ
    all_years_data.append(df_2045_output)

    # 全データを結合
    df_final = pd.concat(all_years_data, ignore_index=True)

    return df_final
