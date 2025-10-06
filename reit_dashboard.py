"""Healthcare REIT Comparative Analysis Dashboard"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="Healthcare REIT Analysis | Portfolio Management",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #0e1117;
    }

    /* Metric styling */
    .stMetric {
        background-color: #1e2530;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3a4a;
    }

    .stMetric label {
        color: #8b9dc3 !important;
        font-weight: 600;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 28px !important;
    }

    .stMetric [data-testid="stMetricDelta"] {
        color: #00ff88 !important;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
    }

    /* Dataframe styling */
    .dataframe {
        background-color: #1e2530 !important;
        color: #ffffff !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
    }

    /* Select boxes and inputs */
    .stSelectbox label, .stRadio label {
        color: #8b9dc3 !important;
    }
</style>
""", unsafe_allow_html=True)

# Data file mappings
DATA_FILES: Dict[str, Path] = {
    "Omega Healthcare (OHI)": Path("/Users/ruthvikmukkamala/Downloads/CF-Export-05-10-2025.xlsx"),
    "CareTrust REIT (CTRE)": Path("/Users/ruthvikmukkamala/Downloads/CF-Export-05-10-2025 (1).xlsx"),
    "Sabra Health Care (SBRA)": Path("/Users/ruthvikmukkamala/Downloads/CF-Export-05-10-2025 (2).xlsx"),
}

# Financial sheets available in the data files
FINANCIAL_SHEETS = [
    "Income Statement",
    "Balance Sheet",
    "Cash Flow",
    "Valuation",
    "Operating Metrics",
]

# =============================================================================
# Data Loading & Processing
# =============================================================================


@st.cache_data
def load_sheet(file_path: Path, sheet_name: str) -> Optional[pd.DataFrame]:
    """Load Excel sheet from file."""
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Failed to load {sheet_name} from {file_path.name}: {str(e)}")
        return None


@st.cache_data
def get_financial_data(file_path: Path, sheet_name: str) -> Optional[pd.DataFrame]:
    """Extract and clean financial data from Excel sheet."""
    df = load_sheet(file_path, sheet_name)
    if df is None:
        return None

    # Extract date headers from row 16 (skip first column)
    date_headers = df.iloc[16, 1:].tolist()
    valid_dates = [str(d) for d in date_headers if pd.notna(d)]

    # Extract data starting from row 17
    df_clean = df.iloc[17:].copy()
    df_clean = df_clean.set_index(df_clean.columns[0])

    # Assign date column names
    df_clean.columns = [
        valid_dates[i] if i < len(valid_dates) else f"Col_{i}"
        for i in range(len(df_clean.columns))
    ]

    # Keep only columns with valid dates
    df_clean = df_clean.iloc[:, : len(valid_dates)]

    # Convert to numeric
    df_clean = df_clean.apply(pd.to_numeric, errors="coerce")

    # Remove empty rows
    df_clean = df_clean.dropna(how="all")
    df_clean.index.name = None

    return df_clean


# =============================================================================
# Helper Functions
# =============================================================================


def format_currency(value: float, scale: str = "M") -> str:
    if pd.isna(value):
        return "N/A"

    scale_map = {"M": 1e6, "B": 1e9, "K": 1e3}
    divisor = scale_map.get(scale, 1)

    if scale in ["M", "B"]:
        return f"${value / divisor:.1f}{scale}"
    return f"${value / divisor:.0f}{scale}"


def safe_extract_value(series: pd.Series, index: int = 0) -> Optional[float]:
    if isinstance(series, pd.Series):
        return series.iloc[index] if len(series) > index else None
    return series if pd.notna(series) else None


def calculate_growth(current: float, previous: float) -> Optional[str]:
    if pd.notna(current) and pd.notna(previous) and previous != 0:
        return f"{((current - previous) / previous * 100):.1f}%"
    return None


# =============================================================================
# UI Configuration
# =============================================================================

st.title("Healthcare REIT Portfolio Analysis")
st.markdown("**Comparative Financial Analysis: OHI | CTRE | SBRA**")
st.markdown("---")

# Sidebar navigation
with st.sidebar:
    st.header("📊 Navigation")
    view = st.radio(
        "Analysis Views",
        [
            "Portfolio Overview",
            "Income Statement",
            "Balance Sheet",
            "Cash Flow",
            "Valuation Metrics",
            "Operating Metrics",
            "Geographic Analysis",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### REITs Tracked")
    st.markdown("• **OHI** - Omega Healthcare")
    st.markdown("• **CTRE** - CareTrust REIT")
    st.markdown("• **SBRA** - Sabra Health Care")
    st.markdown("---")
    st.caption("Data Source: Company Filings")
    st.caption("Updated: October 2025")


# =============================================================================
# Main Content - Views
# =============================================================================

if view == "Portfolio Overview":
    st.header("Portfolio Overview")

    cols = st.columns(3)

    for idx, (name, file_path) in enumerate(DATA_FILES.items()):
        with cols[idx]:
            st.subheader(name.split("(")[1].replace(")", ""))

            # Load financial data
            df_income = get_financial_data(file_path, "Income Statement")
            df_balance = get_financial_data(file_path, "Balance Sheet")
            df_valuation = get_financial_data(file_path, "Valuation")

            if df_income is not None and len(df_income.columns) > 0:
                latest_period = df_income.columns[0]
                st.caption(f"Period: {latest_period}")

                # Revenue metrics
                if "Total revenues" in df_income.index:
                    current_revenue = safe_extract_value(
                        df_income.loc["Total revenues", latest_period]
                    )
                    delta = None

                    if len(df_income.columns) > 1:
                        prev_revenue = safe_extract_value(
                            df_income.loc["Total revenues", df_income.columns[1]]
                        )
                        delta = calculate_growth(current_revenue, prev_revenue)

                    st.metric(
                        "Total Revenue",
                        format_currency(current_revenue, "M"),
                        delta=delta,
                    )

                # Net income metrics
                net_income_key = next(
                    (
                        key
                        for key in [
                            "Net income available for common stockholders",
                            "Net income",
                        ]
                        if key in df_income.index
                    ),
                    None,
                )

                if net_income_key:
                    net_income = safe_extract_value(
                        df_income.loc[net_income_key, latest_period]
                    )
                    st.metric("Net Income", format_currency(net_income, "M"))

            # Balance sheet metrics
            if df_balance is not None and len(df_balance.columns) > 0:
                latest_period = df_balance.columns[0]

                if "Total Assets" in df_balance.index:
                    assets = safe_extract_value(
                        df_balance.loc["Total Assets", latest_period]
                    )
                    st.metric("Total Assets", format_currency(assets, "B"))

            # Valuation metrics
            if df_valuation is not None and len(df_valuation.columns) > 0:
                latest_period = df_valuation.columns[0]
                div_yield_key = "Dividend Yield - Common Stock - Net - Issue Specific - %"

                if div_yield_key in df_valuation.index:
                    div_yield = safe_extract_value(
                        df_valuation.loc[div_yield_key, latest_period]
                    )
                    if pd.notna(div_yield):
                        st.metric("Dividend Yield", f"{div_yield:.2f}%")
                    else:
                        st.metric("Dividend Yield", "N/A")

elif view == "Income Statement":
    st.header("Income Statement Analysis")

    # Metric selection
    income_metrics = [
        "Total revenues",
        "Total expenses",
        "Net income",
        "Rental income",
        "Interest income",
        "Interest expense",
    ]
    selected_metric = st.selectbox("Select Metric", income_metrics)

    # Comparative trend chart
    fig = go.Figure()

    for name, file_path in DATA_FILES.items():
        df = get_financial_data(file_path, "Income Statement")
        if df is not None and selected_metric in df.index:
            values = df.loc[selected_metric].dropna()
            ticker = name.split("(")[1].replace(")", "")

            fig.add_trace(
                go.Scatter(
                    x=values.index,
                    y=values.values / 1e6,  # Convert to millions
                    name=ticker,
                    mode="lines+markers",
                    line=dict(width=2),
                    marker=dict(size=6),
                )
            )

    fig.update_layout(
        title=f"{selected_metric} - Trend Comparison",
        xaxis_title="Period",
        yaxis_title="Amount ($M)",
        hovermode="x unified",
        height=500,
        template="plotly_dark",
        plot_bgcolor='#1e2530',
        paper_bgcolor='#1e2530',
        font=dict(color='#ffffff'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(30, 37, 48, 0.8)',
            bordercolor='#2e3a4a',
            borderwidth=1
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed data table
    st.markdown("### Detailed Financials")
    selected_reit = st.selectbox("Select REIT for Details", list(DATA_FILES.keys()))
    df_detail = get_financial_data(DATA_FILES[selected_reit], "Income Statement")

    if df_detail is not None:
        # Format numbers for display
        df_display = df_detail.copy()
        for col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"${x/1e6:,.1f}M" if pd.notna(x) else "N/A"
            )
        st.dataframe(df_display, use_container_width=True, height=400)

elif view == "Balance Sheet":
    st.header("Balance Sheet Analysis")

    balance_metrics = [
        "Total Assets",
        "Total Liabilities and Stockholders Equity",
        "Total debt",
        "Cash and cash equivalents",
        "Real estate assets - net",
    ]
    selected_metric = st.selectbox("Select Metric", balance_metrics)

    # Latest period comparison
    fig = go.Figure()

    for name, file_path in DATA_FILES.items():
        df = get_financial_data(file_path, "Balance Sheet")
        if df is not None and selected_metric in df.index:
            values = df.loc[selected_metric].dropna()
            ticker = name.split("(")[1].replace(")", "")

            if len(values) > 0:
                latest_value = values.iloc[0] / 1e9  # Convert to billions
                fig.add_trace(
                    go.Bar(
                        x=[ticker],
                        y=[latest_value],
                        name=ticker,
                        text=[f"${latest_value:.2f}B"],
                        textposition="outside",
                    )
                )

    fig.update_layout(
        title=f"{selected_metric} - Latest Period Comparison",
        xaxis_title="REIT",
        yaxis_title="Amount ($B)",
        showlegend=False,
        height=500,
        template="plotly_dark",
        plot_bgcolor='#1e2530',
        paper_bgcolor='#1e2530',
        font=dict(color='#ffffff'),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed data
    st.markdown("### Detailed Balance Sheet")
    selected_reit = st.selectbox("Select REIT for Details", list(DATA_FILES.keys()))
    df_detail = get_financial_data(DATA_FILES[selected_reit], "Balance Sheet")

    if df_detail is not None:
        df_display = df_detail.copy()
        for col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"${x/1e9:,.2f}B" if pd.notna(x) else "N/A"
            )
        st.dataframe(df_display, use_container_width=True, height=400)

elif view == "Cash Flow":
    st.header("Cash Flow Analysis")

    cashflow_metrics = [
        "Net income",
        "Depreciation and amortization",
        "Stock-based compensation expense",
        "Impairment on real estate properties",
    ]
    selected_metric = st.selectbox("Select Metric", cashflow_metrics)

    # Trend comparison
    fig = go.Figure()

    for name, file_path in DATA_FILES.items():
        df = get_financial_data(file_path, "Cash Flow")
        if df is not None and selected_metric in df.index:
            values = df.loc[selected_metric].dropna()
            ticker = name.split("(")[1].replace(")", "")

            fig.add_trace(
                go.Scatter(
                    x=values.index,
                    y=values.values / 1e6,
                    name=ticker,
                    mode="lines+markers",
                    line=dict(width=2),
                    marker=dict(size=6),
                )
            )

    fig.update_layout(
        title=f"{selected_metric} - Historical Trend",
        xaxis_title="Period",
        yaxis_title="Amount ($M)",
        hovermode="x unified",
        height=500,
        template="plotly_dark",
        plot_bgcolor='#1e2530',
        paper_bgcolor='#1e2530',
        font=dict(color='#ffffff'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(30, 37, 48, 0.8)',
            bordercolor='#2e3a4a',
            borderwidth=1
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed data
    st.markdown("### Detailed Cash Flow")
    selected_reit = st.selectbox("Select REIT for Details", list(DATA_FILES.keys()))
    df_detail = get_financial_data(DATA_FILES[selected_reit], "Cash Flow")

    if df_detail is not None:
        df_display = df_detail.copy()
        for col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"${x/1e6:,.1f}M" if pd.notna(x) else "N/A"
            )
        st.dataframe(df_display, use_container_width=True, height=400)

elif view == "Valuation Metrics":
    st.header("Valuation Metrics")

    # Load valuation data for all REITs
    col1, col2, col3 = st.columns(3)

    for idx, (name, file_path) in enumerate(DATA_FILES.items()):
        col = [col1, col2, col3][idx]
        with col:
            st.subheader(name)
            df_val = get_financial_data(file_path, 'Valuation')
            if df_val is not None:
                # Show key valuation metrics
                metrics_to_show = [
                    ('Market Capitalization', 'Market Cap'),
                    ('Enterprise Value', 'EV'),
                    ('Price to Book Value per Share - Issue Specific', 'P/B'),
                    ('Dividend Yield - Common Stock - Net - Issue Specific - %', 'Div Yield')
                ]

                for metric_key, metric_label in metrics_to_show:
                    if metric_key in df_val.index:
                        latest_val = df_val.loc[metric_key].dropna()
                        if len(latest_val) > 0:
                            val = latest_val.iloc[0]
                            if 'Market' in metric_key or 'Enterprise' in metric_key:
                                st.metric(metric_label, f"${val/1e9:.2f}B" if pd.notna(val) else "N/A")
                            else:
                                st.metric(metric_label, f"{val:.2f}" if pd.notna(val) else "N/A")

    # Comparison chart
    st.subheader("Valuation Comparison")
    metric_options = st.multiselect(
        "Select Metrics to Compare:",
        ['Price to Book Value per Share - Issue Specific',
         'Dividend Yield - Common Stock - Net - Issue Specific - %',
         'Market Capitalization',
         'Enterprise Value'],
        default=['Price to Book Value per Share - Issue Specific',
                'Dividend Yield - Common Stock - Net - Issue Specific - %']
    )

    if metric_options:
        for metric in metric_options:
            fig = go.Figure()

            for name, file_path in DATA_FILES.items():
                df = get_financial_data(file_path, 'Valuation')
                if df is not None and metric in df.index:
                    values = df.loc[metric].dropna()
                    fig.add_trace(go.Scatter(
                        x=values.index,
                        y=values.values,
                        name=name,
                        mode='lines+markers'
                    ))

            fig.update_layout(
                title=f"{metric} Over Time",
                xaxis_title="Period",
                yaxis_title=metric,
                hovermode='x unified',
                height=400,
                template="plotly_dark",
                plot_bgcolor='#1e2530',
                paper_bgcolor='#1e2530',
                font=dict(color='#ffffff'),
            )

            st.plotly_chart(fig, use_container_width=True)

elif view == "Operating Metrics":
    st.header("Operating Metrics")

    # Create tabs for each REIT
    tabs = st.tabs(list(DATA_FILES.keys()))

    for idx, (name, file_path) in enumerate(DATA_FILES.items()):
        with tabs[idx]:
            df_ops = get_financial_data(file_path, 'Operating Metrics')
            if df_ops is not None:
                st.dataframe(df_ops, use_container_width=True)

                # Create visualization if numeric data exists
                if len(df_ops) > 0:
                    # Get first few metrics for visualization
                    metrics_to_plot = df_ops.index[:5]

                    for metric in metrics_to_plot:
                        values = df_ops.loc[metric].dropna()
                        if len(values) > 1:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=values.index,
                                y=values.values,
                                mode='lines+markers',
                                name=metric
                            ))
                            fig.update_layout(
                                title=metric,
                                xaxis_title="Period",
                                height=300,
                                template="plotly_dark",
                                plot_bgcolor='#1e2530',
                                paper_bgcolor='#1e2530',
                                font=dict(color='#ffffff'),
                            )
                            st.plotly_chart(fig, use_container_width=True)

elif view == "Geographic Analysis":
    st.header("Geographic Distribution")

    # Only Omega has geographic data
    if 'Omega Healthcare (OHI)' in DATA_FILES:
        st.subheader("Omega Healthcare - Geographic Line By Segment")
        df_geo = load_sheet(DATA_FILES['Omega Healthcare (OHI)'], 'Geographic Line By Se')
        if df_geo is not None:
            st.dataframe(df_geo, use_container_width=True)

        st.subheader("Omega Healthcare - Geographic Line By State")
        df_geo_state = load_sheet(DATA_FILES['Omega Healthcare (OHI)'], 'Geographic Line By St')
        if df_geo_state is not None:
            st.dataframe(df_geo_state, use_container_width=True)

    st.info("Geographic data is primarily available for Omega Healthcare (OHI)")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This dashboard visualizes financial data for three nursing home REITs:
- Omega Healthcare Investors (OHI)
- CareTrust REIT (CTRE)
- Sabra Health Care REIT (SBRA)
""")
