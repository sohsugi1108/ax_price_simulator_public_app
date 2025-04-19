"""TSO（送電事業者）データ分析"""
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# エリアリストの設定
areas = ["Hokkaido", "Tohoku", "Tokyo", "Chubu", "Hokuriku",
         "Kansai", "Chugoku", "Shikoku", "Kyushu"]

# 最小出力閾値の設定
MIN_SOLAR_OUTPUT = 100    # 太陽光の最小平均出力（MW）
MIN_NUCLEAR_OUTPUT = 500  # 原子力の最小平均出力（MW）
MIN_WIND_OUTPUT = 100     # 風力の最小平均出力（MW）
MIN_RATIO_THRESHOLD = 0.01  # 最小比率閾値（%）


def main():
    print("\n=== Analyzing TSO Data ===")

    # JEPX価格データの読み込み
    jepx_file = "../app_data/spot_actual_all.csv"
    df_jepx = pd.read_csv(jepx_file, encoding="shift-jis")

    # 時刻コードを時間に変換
    def convert_time(code):
        hour = (code - 1) // 2
        minute = '00' if code % 2 != 0 else '30'
        return f'{hour:02d}:{minute}'

    df_jepx['time'] = df_jepx['時刻コード'].apply(convert_time)
    df_jepx['datetime'] = pd.to_datetime(
        df_jepx['受渡日'] + ' ' + df_jepx['time'])

    # 2024年度のデータのみを抽出
    start_date = '2024-04-01'
    end_date = '2025-03-31'
    df_jepx = df_jepx[
        (df_jepx['datetime'] >= start_date) &
        (df_jepx['datetime'] <= end_date)
    ]

    # カラム名の変更
    column_mapping = {
        'システムプライス(円/kWh)': 'jepx_System',
        'エリアプライス北海道(円/kWh)': 'Hokkaido',
        'エリアプライス東北(円/kWh)': 'Tohoku',
        'エリアプライス東京(円/kWh)': 'Tokyo',
        'エリアプライス中部(円/kWh)': 'Chubu',
        'エリアプライス北陸(円/kWh)': 'Hokuriku',
        'エリアプライス関西(円/kWh)': 'Kansai',
        'エリアプライス中国(円/kWh)': 'Chugoku',
        'エリアプライス四国(円/kWh)': 'Shikoku',
        'エリアプライス九州(円/kWh)': 'Kyushu'
    }
    df_jepx.rename(columns=column_mapping, inplace=True)
    df_jepx.drop(['時刻コード', '受渡日', 'time'], axis=1, inplace=True)

    results = {
        'solar': {},
        'nuclear': {},
        'wind': {}
    }

    # === Solar Power Analysis ===
    for area in areas:
        print(f"\n=== Analyzing {area} Area (Solar) ===")
        analyze_solar(area, df_jepx, start_date, end_date, results)

    # === Nuclear Power Analysis ===
    valid_nuclear_areas = []
    nuclear_coefficients = []

    for area in areas:
        print(f"\n=== Analyzing {area} Area (Nuclear) ===")
        analyze_nuclear(area, df_jepx, start_date, end_date, results,
                        valid_nuclear_areas, nuclear_coefficients)

    # === Wind Power Analysis ===
    valid_wind_areas = []
    wind_coefficients = []

    for area in areas:
        print(f"\n=== Analyzing {area} Area (Wind) ===")
        analyze_wind(area, df_jepx, start_date, end_date, results,
                     valid_wind_areas, wind_coefficients)

    # 有効なエリアの原子力係数の平均値を計算
    if nuclear_coefficients:
        nuclear_coefficient_mean = np.mean(nuclear_coefficients)
        print(
            f"\n=== Average Nuclear Impact: {nuclear_coefficient_mean:.3f} JPY/kWh ===")
        print(f"Analyzed Areas: {', '.join(valid_nuclear_areas)}")

    # 有効なエリアの風力係数の平均値を計算
    if wind_coefficients:
        wind_coefficient_mean = np.mean(wind_coefficients)
        print(
            f"\n=== Average Wind Impact: {wind_coefficient_mean:.3f} JPY/kWh ===")
        print(f"Analyzed Areas: {', '.join(valid_wind_areas)}")

    save_analysis_summary(results)


def analyze_solar(area, df_jepx, start_date, end_date, results):
    try:
        # TSOデータの読み込み
        tso_file = f"../app_data/tso/tso_{area}.csv"
        if not os.path.exists(tso_file):
            print(f"Warning: {tso_file} not found")
            return

        df_tso = pd.read_csv(tso_file, encoding="utf-8",
                             parse_dates=["datetime"])
        df_tso = df_tso[(df_tso['datetime'] >= start_date) &
                        (df_tso['datetime'] <= end_date)]

        # データの結合と分析
        df_merged = pd.merge(df_jepx, df_tso, on='datetime', how='inner')
        df_merged['solar_ratio'] = df_merged['太陽光発電実績'] / df_merged['合計'] * 100

        # 昼間データの抽出（6:00-18:00）
        df_daytime = df_merged[
            (df_merged['datetime'].dt.hour >= 6) &
            (df_merged['datetime'].dt.hour < 18)
        ]

        # 太陽光出力が小さいデータを除外
        solar_avg = df_daytime['太陽光発電実績'].mean()
        if solar_avg < MIN_SOLAR_OUTPUT:
            print(
                f"Excluded: Average solar output ({solar_avg:.1f}MW) is below threshold")
            return

        # 太陽光比率が小さいデータを除外
        df_daytime = df_daytime[df_daytime['solar_ratio']
                                > MIN_RATIO_THRESHOLD]

        # グラフの作成と保存
        create_analysis_plot(area, df_daytime, 'Solar Power',
                             'solar_ratio', 'Solar Power Ratio (%)')

        # 分析結果の保存
        results['solar'][area] = analyze_and_print_results(
            df_daytime, area, 'Solar', 'solar_ratio')

    except Exception as e:
        print(f"Error analyzing {area}: {str(e)}")


def analyze_nuclear(area, df_jepx, start_date, end_date, results,
                    valid_nuclear_areas, nuclear_coefficients):
    try:
        # TSOデータの読み込み
        tso_file = f"../app_data/tso/tso_{area}.csv"
        if not os.path.exists(tso_file):
            print(f"Warning: {tso_file} not found")
            return

        df_tso = pd.read_csv(tso_file, encoding="utf-8",
                             parse_dates=["datetime"])
        df_tso = df_tso[(df_tso['datetime'] >= start_date) &
                        (df_tso['datetime'] <= end_date)]

        # データの結合と分析
        df_merged = pd.merge(df_jepx, df_tso, on='datetime', how='inner')
        df_merged['nuclear_ratio'] = df_merged['原子力'] / df_merged['合計'] * 100

        # 原子力出力が小さいデータを除外
        nuclear_avg = df_merged['原子力'].mean()
        if nuclear_avg < MIN_NUCLEAR_OUTPUT:
            print(
                f"Excluded: Average nuclear output ({nuclear_avg:.1f}MW) is below threshold")
            return

        # 原子力比率が小さいデータを除外
        df_merged = df_merged[df_merged['nuclear_ratio'] > MIN_RATIO_THRESHOLD]

        # グラフの作成と保存
        create_analysis_plot(area, df_merged, 'Nuclear Power',
                             'nuclear_ratio', 'Nuclear Power Ratio (%)')

        # 分析結果の保存
        results['nuclear'][area] = analyze_and_print_results(
            df_merged, area, 'Nuclear', 'nuclear_ratio')

        valid_nuclear_areas.append(area)
        nuclear_coefficients.append(
            results['nuclear'][area]['all_data']['coefficient'])

    except Exception as e:
        print(f"Error analyzing {area}: {str(e)}")


def analyze_wind(area, df_jepx, start_date, end_date, results,
                 valid_wind_areas, wind_coefficients):
    try:
        # TSOデータの読み込み
        tso_file = f"../app_data/tso/tso_{area}.csv"
        if not os.path.exists(tso_file):
            print(f"Warning: {tso_file} not found")
            return

        df_tso = pd.read_csv(tso_file, encoding="utf-8",
                             parse_dates=["datetime"])
        df_tso = df_tso[(df_tso['datetime'] >= start_date) &
                        (df_tso['datetime'] <= end_date)]

        # データの結合と分析
        df_merged = pd.merge(df_jepx, df_tso, on='datetime', how='inner')
        df_merged['wind_ratio'] = df_merged['風力発電実績'] / df_merged['合計'] * 100

        # 風力出力が小さいデータを除外
        wind_avg = df_merged['風力発電実績'].mean()
        if wind_avg < MIN_WIND_OUTPUT:
            print(
                f"Excluded: Average wind output ({wind_avg:.1f}MW) is below threshold")
            return

        # 風力比率が小さいデータを除外
        df_merged = df_merged[df_merged['wind_ratio'] > MIN_RATIO_THRESHOLD]

        # グラフの作成と保存
        create_analysis_plot(area, df_merged, 'Wind Power',
                             'wind_ratio', 'Wind Power Ratio (%)', 'wind')

        # 分析結果の保存
        results['wind'][area] = analyze_and_print_results(
            df_merged, area, 'Wind', 'wind_ratio')

        valid_wind_areas.append(area)
        wind_coefficients.append(
            results['wind'][area]['all_data']['coefficient'])

    except Exception as e:
        print(f"Error analyzing {area}: {str(e)}")


def create_analysis_plot(area, df, title_prefix, x_column, xlabel, power_type=None):
    X = df[x_column].values.reshape(-1, 1)
    y = df[area].values
    model = LinearRegression()
    model.fit(X, y)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='Price Data')
    plt.plot(X, model.predict(X), color='red',
             linewidth=2, label='Regression Line')

    plt.title(
        f"{title_prefix} Price Impact Analysis\n{area} Area FY2024", pad=20)
    plt.xlabel(xlabel)
    plt.ylabel('JEPX Price (JPY/kWh)')
    plt.grid(True)
    plt.legend()

    # Display regression results
    plt.text(0.05, 0.95,
             f'Price Impact: {model.coef_[0]:.1f} JPY/kWh per 1%\n'
             f'R-squared: {model.score(X, y):.3f}\n'
             f'Data Points: {len(df)}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if power_type is None:
        power_type = 'solar' if x_column == 'solar_ratio' else 'nuclear'
    save_path = f'./result/tso_{area.lower()}_{power_type}_2024.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_and_print_results(df, area, source_name, ratio_column):
    X = df[ratio_column].values.reshape(-1, 1)
    y = df[area].values
    model = LinearRegression()
    model.fit(X, y)

    print(f"\n=== {area} Analysis Results ===")
    print(f"Data Points: {len(df)}")
    print(f"Price Impact per 1% {source_name}: {model.coef_[0]:.3f} JPY/kWh")
    print(f"R-squared: {model.score(X, y):.3f}")

    return {
        'all_data': {
            'samples': len(df),
            'coefficient': model.coef_[0],
            'r2': model.score(X, y)
        }
    }


def save_analysis_summary(results):
    # Solar Power Summary
    solar_summary_data = []
    for area, result in results['solar'].items():
        solar_summary_data.append({
            'Area': area,
            'Samples': result['all_data']['samples'],
            'Price_Impact': result['all_data']['coefficient'],
            'R2': result['all_data']['r2']
        })
    pd.DataFrame(solar_summary_data).to_csv(
        './result/solar_regression_summary_2024.csv', index=False)

    # Nuclear Power Summary
    nuclear_summary_data = []
    for area, result in results['nuclear'].items():
        nuclear_summary_data.append({
            'Area': area,
            'Samples': result['all_data']['samples'],
            'Price_Impact': result['all_data']['coefficient'],
            'R2': result['all_data']['r2']
        })
    pd.DataFrame(nuclear_summary_data).to_csv(
        './result/nuclear_regression_summary_2024.csv', index=False)

    # Wind Power Summary
    wind_summary_data = []
    for area, result in results['wind'].items():
        wind_summary_data.append({
            'Area': area,
            'Samples': result['all_data']['samples'],
            'Price_Impact': result['all_data']['coefficient'],
            'R2': result['all_data']['r2']
        })
    pd.DataFrame(wind_summary_data).to_csv(
        './result/wind_regression_summary_2024.csv', index=False)


if __name__ == "__main__":
    main()
