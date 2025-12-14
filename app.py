import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Cash Flow Functions
# ---------------------------------------------------------

def project_cash_flow(
        monthly_costs,
        jobs_per_week,
        avg_revenue_per_job,
        seasonality_factors,
        annual_growth_rate,
        horizon_years
):
    months = horizon_years * 12
    data = []

    for m in range(months):
        month_index = m % 12
        year_index = m // 12

        growth_multiplier = (1 + annual_growth_rate) ** year_index
        season = seasonality_factors[month_index]

        monthly_revenue = (
                jobs_per_week * 4 * avg_revenue_per_job *
                season * growth_multiplier
        )
        monthly_profit = monthly_revenue - monthly_costs

        data.append([
            m + 1,
            year_index + 1,
            monthly_revenue,
            monthly_costs,
            monthly_profit
        ])

    df = pd.DataFrame(data, columns=[
        "Month", "Year", "Revenue", "Costs", "Profit"
    ])
    df["Cumulative Profit"] = df["Profit"].cumsum()

    return df


def monte_carlo_cash_flow(
        monthly_costs,
        jobs_per_week,
        avg_revenue_per_job,
        seasonality_factors,
        annual_growth_rate,
        horizon_years,
        simulations=2000
):
    months = horizon_years * 12
    results = np.zeros((simulations, months))

    for m in range(months):
        year_index = m // 12
        month_index = m % 12

        season = seasonality_factors[month_index]
        growth = (1 + annual_growth_rate) ** year_index

        mean_rev = jobs_per_week * 4 * avg_revenue_per_job * season * growth
        std_rev = avg_revenue_per_job * 0.15

        rev_samples = np.random.normal(loc=mean_rev, scale=std_rev, size=simulations)
        rev_samples = np.maximum(0, rev_samples)

        results[:, m] = rev_samples - monthly_costs

    summary = pd.DataFrame({
        "Month": np.arange(1, months + 1),
        "Median Profit": np.median(results, axis=0),
        "Lower_5pct": np.percentile(results, 5, axis=0),
        "Upper_95pct": np.percentile(results, 95, axis=0)
    })

    return summary


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.title("Power Washing Business Cash Flow Projection")
st.markdown("### 5-Year Forecasting Tool with Optional Uncertainty Modeling")

st.sidebar.header("Business Inputs")

# ------------------- Cost Inputs -------------------------
st.sidebar.subheader("Startup Costs (one-time)")
equipment = st.sidebar.number_input("Equipment", 500, 10000, 2000)
marketing_startup = st.sidebar.number_input("Startup Marketing", 0, 5000, 400)
legal = st.sidebar.number_input("Legal/LLC/Insurance Setup", 0, 3000, 800)
misc = st.sidebar.number_input("Misc", 0, 2000, 200)

startup_costs = equipment + marketing_startup + legal + misc

st.sidebar.subheader("Monthly Operating Costs")
fuel = st.sidebar.number_input("Fuel", 0, 1000, 120)
chemicals = st.sidebar.number_input("Chemicals", 0, 1000, 80)
insurance = st.sidebar.number_input("Insurance", 0, 1000, 90)
marketing_monthly = st.sidebar.number_input("Marketing (ads)", 0, 3000, 300)
software = st.sidebar.number_input("Software/Website", 0, 2000, 50)
maintenance = st.sidebar.number_input("Maintenance", 0, 1000, 40)

monthly_costs = fuel + chemicals + insurance + marketing_monthly + software + maintenance

# ------------------- Revenue Inputs ----------------------
st.sidebar.header("Revenue Model")
jobs_per_week = st.sidebar.number_input("Jobs per Week", 1, 100, 10)
avg_revenue = st.sidebar.number_input("Avg Revenue per Job ($)", 50, 2000, 180)
annual_growth = st.sidebar.slider("Annual Growth Rate (%)", 0.0, 0.50, 0.07)
horizon_years = st.sidebar.slider("Projection Horizon (Years)", 1, 10, 5)

# ------------------- Seasonality -------------------------
st.sidebar.header("Seasonality (per month multiplier)")
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

seasonality_factors = []
for m in months:
    seasonality_factors.append(
        st.sidebar.slider(m, 0.5, 1.5, 1.0)
    )

# ------------------- Monte Carlo Toggle -------------------
use_monte_carlo = st.sidebar.checkbox("Enable Monte-Carlo Forecasting")

# ---------------------------------------------------------
# Main Output
# ---------------------------------------------------------

st.subheader("Startup Cost Summary")
st.write(f"**Total Startup Costs:** ${startup_costs:,.2f}")

st.subheader("Monthly Operating Cost Summary")
st.write(f"**Monthly Operating Costs:** ${monthly_costs:,.2f}")

# ---------------------------------------------------------
# Deterministic Projection
# ---------------------------------------------------------
df = project_cash_flow(
    monthly_costs=monthly_costs,
    jobs_per_week=jobs_per_week,
    avg_revenue_per_job=avg_revenue,
    seasonality_factors=seasonality_factors,
    annual_growth_rate=annual_growth,
    horizon_years=horizon_years
)

st.subheader("5-Year Cash Flow Projection (Deterministic)")
st.dataframe(df.head(24))  # first 2 years

# Plot deterministic
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Month"], df["Profit"], label="Profit")
ax.plot(df["Month"], df["Cumulative Profit"], label="Cumulative Profit")
ax.set_title("Deterministic Profit Projection")
ax.set_xlabel("Month")
ax.set_ylabel("USD")
ax.legend()
st.pyplot(fig)

# ---------------------------------------------------------
# Monte-Carlo Simulation
# ---------------------------------------------------------
if use_monte_carlo:
    st.subheader("Monte-Carlo Profit Projection (Uncertainty Included)")

    summary = monte_carlo_cash_flow(
        monthly_costs=monthly_costs,
        jobs_per_week=jobs_per_week,
        avg_revenue_per_job=avg_revenue,
        seasonality_factors=seasonality_factors,
        annual_growth_rate=annual_growth,
        horizon_years=horizon_years
    )

    st.write("95% Confidence Interval per month:")
    st.dataframe(summary.head(24))

    # Plot uncertainty bands
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(summary["Month"], summary["Median Profit"], label="Median Profit")
    ax2.fill_between(
        summary["Month"],
        summary["Lower_5pct"],
        summary["Upper_95pct"],
        alpha=0.3,
        label="5%–95% range"
    )
    ax2.set_title("Monte-Carlo Profit Projection (Uncertainty)")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("USD")
    ax2.legend()
    st.pyplot(fig2)

# ---------------------------------------------------------
# Download CSV
# ---------------------------------------------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Cash Flow Data as CSV",
    csv,
    "cash_flow_projection.csv",
    "text/csv"
)