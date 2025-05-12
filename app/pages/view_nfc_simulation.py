import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Load data
data = pd.read_csv("app_data/nfc_actual.csv")
icp_data = pd.read_csv("app_data/icp_prospect.csv")


def calculate_cagr(start_value, end_value, periods):
    return (end_value / start_value) ** (1 / periods) - 1

# Extend data to 2050 based on CAGR


def extend_data(data, demand_cagr, supply_cagr):
    years = list(range(2024, 2051))
    demand = [data["nfc_demand"].iloc[-1]]
    supply = [data["nfc_supply"].iloc[-1]]

    for _ in years[1:]:
        demand.append(demand[-1] * (1 + demand_cagr))
        supply.append(supply[-1] * (1 + supply_cagr))

    return pd.DataFrame({"year": years, "demand": demand, "supply": supply})


def get_adjusted_icp_values(carbon_price):
    """ICPの推移を計算する"""
    # 2050年の値に対する調整係数を計算
    scale_factor = carbon_price / icp_data['price'].iloc[-1]
    # 2024年の値を基準とした比率に調整係数を適用
    base_value = icp_data['price'].iloc[0]
    adjusted_values = [(v / base_value) * scale_factor *
                       base_value for v in icp_data['price']]
    return pd.DataFrame({'year': icp_data['year'], 'price': adjusted_values})

# Simulation logic


def simulate(demand_cagr, supply_cagr, emission_factor, carbon_price):
    extended_data = extend_data(data, demand_cagr, supply_cagr)
    extended_data["price"] = 0.4  # Default price

    for i, row in extended_data.iterrows():
        if row["demand"] > row["supply"]:
            extended_data.at[i, "price"] = emission_factor * carbon_price

    return extended_data


def create_nfc_simulation_view():
    """非化石証書価格シミュレーションのビューを作成"""
    st.title("非化石証書価格シミュレーション")

    # Layout with three columns
    col1, col2, col3 = st.columns([1, 1, 1])

    # All sliders in first column with comments
    with col1:
        # Demand CAGR slider with comments
        st.markdown("###### 非化石証書証書　需要成長率（CAGR,%）")
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between;">
                <span>低位</span>
                <span>21-23実績(112.7%)</span>
                <span>高位</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        demand_cagr = st.slider("CAGR(%)",
                                0.0, 200.0, 112.7, key="demand_cagr")/100

        # Supply CAGR slider with comments
        st.markdown("###### 非化石証書証書　供給成長率（CAGR,%）")
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between;">
                <span>低位</span>
                <span>21-23実績(2.3%)</span>
                <span>高位</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        supply_cagr = st.slider("CAGR(%)",
                                0.0, 200.0, 0.0, key="supply_cagr")/100

        # Emission factor slider with comments
        st.markdown("###### 排出係数 2050年想定 (kg-CO2/kWh)")
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between;">
                <span>排出ゼロ</span>
                <span>中庸</span>
                <span>24実績</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        emission_factor = st.slider(
            "排出係数", 0.0, 0.438, 0.0, key="emission_factor") / 1000

        # Carbon price slider with comments
        st.markdown("###### ICP(Internal Carbon Pricing) 2050年想定 (円/t-CO2)")
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between;">
                <span>国内中央値</span>
                <span>先進国NetZero(31000)</span>
                <span>高位</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        carbon_price = st.slider(
            "ICP", 5000, 40000, 35000, key="carbon_price")

    # Run simulation after all sliders are defined
    result = simulate(demand_cagr, supply_cagr, emission_factor, carbon_price)
    emission_years = list(range(2024, 2051))

    # Calculate linear changes
    emission_values = [
        0.000438 + (emission_factor - 0.000438) * (year - 2024) / (2050 - 2024)
        for year in emission_years
    ]

    # Get adjusted ICP values
    adjusted_icp = get_adjusted_icp_values(carbon_price)

    # Calculate NFC price with adjusted ICP values
    nfc_prices = []
    for i, row in result.iterrows():
        year = row["year"]
        year_emission = emission_values[i]
        year_carbon = adjusted_icp.loc[adjusted_icp['year']
                                       == year, 'price'].iloc[0]

        # Calculate price based on emission and carbon values, with 0.4 minimum
        if row["demand"] > row["supply"]:
            price = max(0.4, year_emission * year_carbon)
        else:
            price = 0.4
        nfc_prices.append(price)

    # Display graphs in second and third columns
    with col2:
        # Demand and Supply graph
        fig_demand_supply = go.Figure()
        fig_demand_supply.add_trace(
            go.Bar(x=result["year"], y=result["demand"]/1000000, name="需要", marker_color="lightblue"))
        fig_demand_supply.add_trace(
            go.Bar(x=result["year"], y=result["supply"]/1000000, name="供給", marker_color="lightgreen"))
        fig_demand_supply.update_layout(
            title=dict(
                text="非化石証書需給（GWh,対数表示）",
                xanchor="center",
                x=0.5,
                y=0.95
            ),
            barmode="group",
            height=350,
            yaxis_type="log",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.8,
                xanchor="center",
                x=0.5
            ))
        st.plotly_chart(fig_demand_supply, use_container_width=True)

    with col3:
        # ICP trend graph
        fig_icp = go.Figure()
        fig_icp.add_trace(
            go.Scatter(x=adjusted_icp["year"], y=adjusted_icp["price"],
                       mode="lines", name="ICP推移"))
        fig_icp.update_layout(
            title=dict(
                text="ICP推移 (円/t-CO2)",
                xanchor="center",
                x=0.5,
                y=0.95
            ),
            height=350,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            yaxis=dict(range=[0, 40000])
        )
        st.plotly_chart(fig_icp, use_container_width=True)

        # NFC price graph
        fig_nfc_price = go.Figure()
        fig_nfc_price.add_trace(
            go.Scatter(x=result["year"], y=nfc_prices, mode="lines", name="非化石証書価格"))
        fig_nfc_price.update_layout(
            title=dict(
                text="非化石証書価格推移 (円/kWh)",
                xanchor="center",
                x=0.5,
                y=0.95
            ),
            height=350,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            yaxis=dict(range=[0, 6.0])
        )
        st.plotly_chart(fig_nfc_price, use_container_width=True)
