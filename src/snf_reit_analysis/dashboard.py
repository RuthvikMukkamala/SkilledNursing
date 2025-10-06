"""Comprehensive Streamlit dashboard for SNF REIT analysis."""

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Healthcare REIT Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Metric cards */
    .stMetric {
        background-color: #1e2530;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3a4a;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .stMetric label {
        color: #8b9dc3 !important;
        font-weight: 600;
        font-size: 14px;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 700;
    }

    .stMetric [data-testid="stMetricDelta"] {
        color: #00ff88 !important;
    }

    /* Headers */
    h1 {
        color: #ffffff !important;
        font-weight: 700;
    }

    h2 {
        color: #ffffff !important;
        font-weight: 600;
    }

    h3 {
        color: #e0e0e0 !important;
        font-weight: 600;
    }

    /* Dataframe */
    .dataframe {
        background-color: #1e2530 !important;
    }

    .dataframe th {
        background-color: #2e3a4a !important;
        color: #ffffff !important;
        font-weight: 600;
    }

    .dataframe td {
        background-color: #1e2530 !important;
        color: #e0e0e0 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    /* Form elements */
    .stSelectbox label, .stRadio label, .stMultiselect label {
        color: #8b9dc3 !important;
        font-weight: 500;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1e2530;
        border-color: #2e3a4a;
        color: #ffffff;
    }

    /* Info/Warning/Success boxes */
    .stAlert {
        background-color: #1e2530 !important;
        border: 1px solid #2e3a4a;
    }
</style>
""", unsafe_allow_html=True)

# Data loading
@st.cache_data(ttl=3600)
def load_data(file_name: str) -> Optional[pd.DataFrame]:
    """Load data from processed CSV files."""
    try:
        file_path = Path("data/processed") / file_name
        if file_path.exists():
            return pd.read_csv(file_path)
        return None
    except Exception as e:
        st.error(f"Error loading {file_name}: {e}")
        return None

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Analysis:",
    ["Overview", "CMS Nursing Homes", "REIT Financials", "REIT Comparison", "Actual REIT Properties", "Deep Dive Analysis", "BLS Economic Data", "Data Explorer"]
)

# Main title
st.title("Healthcare REIT Analysis Platform")
st.markdown("*Professional analysis of CMS nursing home data and healthcare REIT performance*")
st.markdown("---")

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "Overview":
    st.header("Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    # Load CMS data for metrics
    cms_data = load_data("cms_provider_info_latest.csv")

    with col1:
        st.subheader("CMS Data")
        if cms_data is not None:
            st.metric("Total Nursing Homes", f"{len(cms_data):,}")
            st.metric("Avg Overall Rating", f"{cms_data['overall_rating'].astype(float).mean():.2f} / 5")
        else:
            st.warning("CMS data not available")

    with col2:
        st.subheader("REIT Data")
        st.metric("REITs Tracked", "3")
        st.metric("Latest Quarter", "Q2 2025")

    with col3:
        st.subheader("Data Coverage")
        st.metric("States Covered", "All 50 States")
        st.metric("Historical Data", "Back to 2010")

    st.markdown("---")

    # Quick stats
    st.subheader("Latest REIT Performance (Q2 2025)")

    reit_data = {
        'REIT': ['OHI (Omega Healthcare)', 'CTRE (CareTrust)', 'SBRA (Sabra)'],
        'Q2 2025 Revenue': ['$282.5M', '$112.5M', '$189.2M'],
        'Total Assets': ['$10.55B', '$4.66B', '$5.33B'],
        'YTD Revenue': ['$559.3M', '$209.1M', '$372.7M']
    }

    st.dataframe(pd.DataFrame(reit_data), use_container_width=True)

    st.markdown("---")
    st.info("Use the sidebar to navigate to different analysis sections")

# ============================================================================
# PAGE 2: CMS NURSING HOMES
# ============================================================================
elif page == "CMS Nursing Homes":
    st.header("CMS Nursing Home Analysis")

    cms_data = load_data("cms_provider_info_latest.csv")

    if cms_data is None:
        st.error("CMS data not available. Run `make data-cms` to fetch data.")
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Facilities", f"{len(cms_data):,}")
        with col2:
            avg_rating = cms_data['overall_rating'].astype(float).mean()
            st.metric("Avg Overall Rating", f"{avg_rating:.2f} / 5")
        with col3:
            total_beds = cms_data['number_of_certified_beds'].astype(float).sum()
            st.metric("Total Certified Beds", f"{total_beds:,.0f}")
        # with col4:
        #     states = cms_data['state'].nunique()
        #     st.metric("States Covered", states)

        st.markdown("---")

        # State selection
        states = sorted(cms_data['state'].unique())
        selected_state = st.selectbox("Filter by State:", ["All States"] + states)

        if selected_state != "All States":
            cms_data = cms_data[cms_data['state'] == selected_state]

        # Rating distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Overall Rating Distribution")
            rating_counts = cms_data['overall_rating'].value_counts().sort_index()
            fig = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={'x': 'Rating', 'y': 'Number of Facilities'},
                title=f"Rating Distribution ({selected_state})",
                color_discrete_sequence=['#00D9FF']
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor='#1e2530',
                paper_bgcolor='#1e2530',
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Top 10 States by Facilities")
            state_counts = cms_data['state'].value_counts().head(10)
            fig = px.bar(
                x=state_counts.values,
                y=state_counts.index,
                orientation='h',
                labels={'x': 'Number of Facilities', 'y': 'State'},
                color_discrete_sequence=['#FF6B9D']
            )
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor='#1e2530',
                paper_bgcolor='#1e2530',
                font=dict(color='#ffffff')
            )
            st.plotly_chart(fig, use_container_width=True)

        # Ownership type
        st.subheader("Ownership Type Distribution")
        ownership = cms_data['ownership_type'].value_counts()
        fig = px.pie(
            values=ownership.values,
            names=ownership.index,
            title="Facilities by Ownership Type",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='#1e2530',
            paper_bgcolor='#1e2530',
            font=dict(color='#ffffff')
        )
        st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.subheader("Facility Data")
        display_cols = ['provider_name', 'state', 'overall_rating', 'health_inspection_rating',
                       'staffing_rating', 'qm_rating', 'number_of_certified_beds']
        st.dataframe(cms_data[display_cols].head(100), use_container_width=True)

# ============================================================================
# PAGE 3: REIT FINANCIALS
# ============================================================================
elif page == "REIT Financials":
    st.header("REIT Financial Analysis")

    # REIT selection
    reit = st.selectbox("Select REIT:", ["OHI", "CTRE", "SBRA"])

    reit_names = {
        "OHI": "Omega Healthcare Investors",
        "CTRE": "CareTrust REIT",
        "SBRA": "Sabra Health Care REIT"
    }

    st.subheader(f"{reit} - {reit_names[reit]}")

    # Load revenue data
    rev_data = load_data(f"sec_{reit.lower()}_revenues_latest.csv")
    assets_data = load_data(f"sec_{reit.lower()}_assets_latest.csv")

    if rev_data is None:
        st.error(f"Data not available for {reit}. Run `make data-sec` to fetch data.")
    else:
        # Filter for quarterly data
        rev_data['end'] = pd.to_datetime(rev_data['end'])
        rev_data = rev_data.sort_values('end', ascending=False)

        # Latest metrics
        latest_q = rev_data[rev_data['fp'].str.startswith('Q')].iloc[0] if len(rev_data[rev_data['fp'].str.startswith('Q')]) > 0 else None

        if latest_q is not None:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Latest Quarter Revenue",
                    f"${latest_q['val']/1e6:.1f}M",
                    f"{latest_q['fy']} {latest_q['fp']}"
                )

            if assets_data is not None:
                assets_data['end'] = pd.to_datetime(assets_data['end'])
                latest_assets = assets_data.iloc[0]

                with col2:
                    st.metric(
                        "Total Assets",
                        f"${latest_assets['val']/1e9:.2f}B",
                        latest_assets['end'].strftime('%Y-%m-%d')
                    )

        st.markdown("---")

        # Revenue trend
        st.subheader("Quarterly Revenue Trend")

        # Filter for quarterly reports only
        quarterly = rev_data[rev_data['fp'].str.startswith('Q')].copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=quarterly['end'],
            y=quarterly['val'] / 1e6,
            mode='lines+markers',
            name='Revenue',
            line=dict(width=3, color='#00D9FF'),
            marker=dict(size=8, color='#00D9FF')
        ))

        fig.update_layout(
            xaxis_title="Quarter End Date",
            yaxis_title="Revenue (Millions USD)",
            hovermode='x unified',
            height=400,
            template="plotly_dark",
            plot_bgcolor='#1e2530',
            paper_bgcolor='#1e2530',
            font=dict(color='#ffffff')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Assets trend
        if assets_data is not None:
            st.subheader("Total Assets Trend")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=assets_data['end'],
                y=assets_data['val'] / 1e9,
                mode='lines+markers',
                name='Assets',
                line=dict(width=3, color='#00FF88'),
                marker=dict(size=8, color='#00FF88')
            ))

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Total Assets (Billions USD)",
                hovermode='x unified',
                height=400,
                template="plotly_dark",
                plot_bgcolor='#1e2530',
                paper_bgcolor='#1e2530',
                font=dict(color='#ffffff')
            )

            st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.subheader("Revenue Data")
        display_data = quarterly[['end', 'val', 'fy', 'fp', 'form', 'filed']].head(20)
        display_data['val'] = display_data['val'].apply(lambda x: f"${x/1e6:.2f}M")
        st.dataframe(display_data, use_container_width=True)

# ============================================================================
# PAGE 4: REIT COMPARISON
# ============================================================================
elif page == "REIT Comparison":
    st.header("REIT Comparison Analysis")

    # Load all REIT data
    ohi_rev = load_data("sec_ohi_revenues_latest.csv")
    ctre_rev = load_data("sec_ctre_revenues_latest.csv")
    sbra_rev = load_data("sec_sbra_revenues_latest.csv")

    if all([ohi_rev is not None, ctre_rev is not None, sbra_rev is not None]):
        # Combine data
        ohi_rev['REIT'] = 'OHI'
        ctre_rev['REIT'] = 'CTRE'
        sbra_rev['REIT'] = 'SBRA'

        all_rev = pd.concat([ohi_rev, ctre_rev, sbra_rev])
        all_rev['end'] = pd.to_datetime(all_rev['end'])

        # Filter for quarterly only
        all_rev = all_rev[all_rev['fp'].str.startswith('Q')].copy()
        all_rev = all_rev.sort_values('end')

        # Revenue comparison
        st.subheader("Quarterly Revenue Comparison")

        fig = go.Figure()

        colors_map = {
            'OHI': '#00D9FF',
            'CTRE': '#FF6B9D',
            'SBRA': '#00FF88'
        }

        for reit in ['OHI', 'CTRE', 'SBRA']:
            reit_data = all_rev[all_rev['REIT'] == reit]
            fig.add_trace(go.Scatter(
                x=reit_data['end'],
                y=reit_data['val'] / 1e6,
                mode='lines+markers',
                name=reit,
                line=dict(width=3, color=colors_map[reit]),
                marker=dict(size=8, color=colors_map[reit])
            ))

        fig.update_layout(
            xaxis_title="Quarter End Date",
            yaxis_title="Revenue (Millions USD)",
            hovermode='x unified',
            height=500,
            template="plotly_dark",
            plot_bgcolor='#1e2530',
            paper_bgcolor='#1e2530',
            font=dict(color='#ffffff'),
            legend=dict(
                bgcolor='rgba(30, 37, 48, 0.8)',
                bordercolor='#2e3a4a',
                borderwidth=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Latest quarter comparison
        st.subheader("Latest Quarter Revenue (Q2 2025)")

        latest = all_rev[all_rev['end'] == all_rev['end'].max()]

        fig = px.bar(
            latest,
            x='REIT',
            y='val',
            color='REIT',
            labels={'val': 'Revenue (USD)'},
            title='Q2 2025 Revenue by REIT',
            color_discrete_map=colors_map
        )

        fig.update_yaxes(tickformat="$,.0f")
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='#1e2530',
            paper_bgcolor='#1e2530',
            font=dict(color='#ffffff')
        )
        st.plotly_chart(fig, use_container_width=True)

        # Market share
        st.subheader("Market Share (Q2 2025 Revenue)")

        market_share = latest.groupby('REIT')['val'].sum()

        fig = px.pie(
            values=market_share.values,
            names=market_share.index,
            title='Revenue Market Share',
            color_discrete_map=colors_map
        )

        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='#1e2530',
            paper_bgcolor='#1e2530',
            font=dict(color='#ffffff')
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Not all REIT data is available. Run `make data-sec` to fetch data.")

# ============================================================================
# PAGE 5: DEEP DIVE ANALYSIS
# ============================================================================
elif page == "Deep Dive Analysis":
    st.header("Comprehensive SEC Financial Deep Dive")
    st.markdown("*Detailed comparative analysis across all key financial metrics*")
    st.markdown("---")

    # Load all REIT data
    ohi_rev = load_data("sec_ohi_revenues_latest.csv")
    ctre_rev = load_data("sec_ctre_revenues_latest.csv")
    sbra_rev = load_data("sec_sbra_revenues_latest.csv")

    ohi_assets = load_data("sec_ohi_assets_latest.csv")
    ctre_assets = load_data("sec_ctre_assets_latest.csv")
    sbra_assets = load_data("sec_sbra_assets_latest.csv")

    ohi_liab = load_data("sec_ohi_liabilities_latest.csv")
    ctre_liab = load_data("sec_ctre_liabilities_latest.csv")
    sbra_liab = load_data("sec_sbra_liabilities_latest.csv")

    ohi_equity = load_data("sec_ohi_stockholdersequity_latest.csv")
    ctre_equity = load_data("sec_ctre_stockholdersequity_latest.csv")
    sbra_equity = load_data("sec_sbra_stockholdersequity_latest.csv")

    ohi_ni = load_data("sec_ohi_netincomeloss_latest.csv")
    ctre_ni = load_data("sec_ctre_netincomeloss_latest.csv")
    sbra_ni = load_data("sec_sbra_netincomeloss_latest.csv")

    ohi_debt = load_data("sec_ohi_longtermdebt_latest.csv")
    ctre_debt = load_data("sec_ctre_longtermdebt_latest.csv")
    sbra_debt = load_data("sec_sbra_longtermdebt_latest.csv")

    ohi_re = load_data("sec_ohi_realestateinvestmentpropertynet_latest.csv")
    ctre_re = load_data("sec_ctre_realestateinvestmentpropertynet_latest.csv")
    sbra_re = load_data("sec_sbra_realestateinvestmentpropertynet_latest.csv")

    # Check if data is available
    if all([ohi_rev is not None, ctre_rev is not None, sbra_rev is not None]):

        # ====================================================================
        # SECTION 1: SIZE & SCALE
        # ====================================================================
        st.subheader("1. Size & Scale Comparison")

        # Calculate size metrics
        size_metrics = []

        for reit_name, rev_df, assets_df, re_df in [
            ('OHI', ohi_rev, ohi_assets, ohi_re),
            ('CTRE', ctre_rev, ctre_assets, ctre_re),
            ('SBRA', sbra_rev, sbra_assets, sbra_re)
        ]:
            # Latest assets
            latest_assets = assets_df.iloc[0]['val'] if len(assets_df) > 0 else None

            # Latest real estate
            latest_re = re_df.iloc[0]['val'] if len(re_df) > 0 else None

            # TTM revenue
            quarterly = rev_df[rev_df['fp'].str.startswith('Q')].copy()
            quarterly = quarterly.sort_values('end', ascending=False)
            ttm_revenue = quarterly.iloc[0:4]['val'].sum() if len(quarterly) >= 4 else None

            size_metrics.append({
                'REIT': reit_name,
                'Total Assets ($B)': latest_assets / 1e9 if latest_assets else 0,
                'Real Estate ($B)': latest_re / 1e9 if latest_re else 0,
                'TTM Revenue ($M)': ttm_revenue / 1e6 if ttm_revenue else 0
            })

        size_df = pd.DataFrame(size_metrics)

        # Display metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Largest by Assets",
                     size_df.loc[size_df['Total Assets ($B)'].idxmax(), 'REIT'],
                     f"${size_df['Total Assets ($B)'].max():.2f}B")

        with col2:
            st.metric("Largest by Real Estate",
                     size_df.loc[size_df['Real Estate ($B)'].idxmax(), 'REIT'],
                     f"${size_df['Real Estate ($B)'].max():.2f}B")

        with col3:
            st.metric("Largest by Revenue",
                     size_df.loc[size_df['TTM Revenue ($M)'].idxmax(), 'REIT'],
                     f"${size_df['TTM Revenue ($M)'].max():.1f}M")

        # Visualize
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Total Assets', 'Real Estate Portfolio', 'TTM Revenue')
        )

        fig.add_trace(go.Bar(x=size_df['REIT'], y=size_df['Total Assets ($B)'],
                            marker_color='#00D9FF', showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=size_df['REIT'], y=size_df['Real Estate ($B)'],
                            marker_color='#00FF88', showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=size_df['REIT'], y=size_df['TTM Revenue ($M)'],
                            marker_color='#FF6B9D', showlegend=False), row=1, col=3)

        fig.update_layout(
            height=400,
            title_text="Size Comparison",
            template="plotly_dark",
            plot_bgcolor='#1e2530',
            paper_bgcolor='#1e2530',
            font=dict(color='#ffffff')
        )
        fig.update_yaxes(title_text="Billions USD", row=1, col=1)
        fig.update_yaxes(title_text="Billions USD", row=1, col=2)
        fig.update_yaxes(title_text="Millions USD", row=1, col=3)

        st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.dataframe(size_df.set_index('REIT').T, use_container_width=True)

        st.markdown("---")

        # ====================================================================
        # SECTION 2: GROWTH ANALYSIS
        # ====================================================================
        st.subheader("2. Revenue Growth & Momentum")

        # Calculate growth metrics
        growth_metrics = []

        for reit_name, rev_df in [('OHI', ohi_rev), ('CTRE', ctre_rev), ('SBRA', sbra_rev)]:
            quarterly = rev_df[rev_df['fp'].str.startswith('Q')].copy()
            quarterly = quarterly.sort_values('end', ascending=False)

            if len(quarterly) >= 8:
                # Latest TTM
                latest_ttm = quarterly.iloc[0:4]['val'].sum()

                # Prior TTM
                prior_ttm = quarterly.iloc[4:8]['val'].sum()

                # YoY growth
                yoy_growth = ((latest_ttm / prior_ttm) - 1) * 100

                # 3-year CAGR
                if len(quarterly) >= 16:
                    ttm_3y_ago = quarterly.iloc[12:16]['val'].sum()
                    cagr_3y = (((latest_ttm / ttm_3y_ago) ** (1/3)) - 1) * 100
                else:
                    cagr_3y = None

                # Revenue volatility
                quarterly['qoq_growth'] = quarterly['val'].pct_change(-1) * 100
                volatility = quarterly['qoq_growth'].std()

                growth_metrics.append({
                    'REIT': reit_name,
                    'TTM Revenue ($M)': latest_ttm / 1e6,
                    'YoY Growth (%)': yoy_growth,
                    '3Y CAGR (%)': cagr_3y if cagr_3y else 0,
                    'Revenue Volatility (%)': volatility
                })

        growth_df = pd.DataFrame(growth_metrics)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            fastest = growth_df.loc[growth_df['YoY Growth (%)'].idxmax(), 'REIT']
            st.metric("Fastest YoY Growth", fastest,
                     f"{growth_df['YoY Growth (%)'].max():.2f}%")

        with col2:
            strongest_3y = growth_df.loc[growth_df['3Y CAGR (%)'].idxmax(), 'REIT']
            st.metric("Strongest 3Y CAGR", strongest_3y,
                     f"{growth_df['3Y CAGR (%)'].max():.2f}%")

        with col3:
            most_stable = growth_df.loc[growth_df['Revenue Volatility (%)'].idxmin(), 'REIT']
            st.metric("Most Stable", most_stable,
                     f"{growth_df['Revenue Volatility (%)'].min():.2f}% volatility")

        with col4:
            avg_growth = growth_df['YoY Growth (%)'].mean()
            st.metric("Average Growth", f"{avg_growth:.2f}%", "All 3 REITs")

        # Visualize growth
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Growth Rates', 'Revenue Stability')
        )

        colors_list = ['#00D9FF', '#FF6B9D']

        for idx, col in enumerate(['YoY Growth (%)', '3Y CAGR (%)']):
            fig.add_trace(go.Bar(x=growth_df['REIT'], y=growth_df[col],
                                name=col, marker_color=colors_list[idx]), row=1, col=1)

        fig.add_trace(go.Bar(x=growth_df['REIT'], y=growth_df['Revenue Volatility (%)'],
                            marker_color='#FFA500', showlegend=False), row=1, col=2)

        fig.update_layout(
            height=400,
            title_text="Growth & Stability Analysis",
            template="plotly_dark",
            plot_bgcolor='#1e2530',
            paper_bgcolor='#1e2530',
            font=dict(color='#ffffff')
        )
        fig.update_yaxes(title_text="Growth Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Std Dev (%)", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

        # Historical revenue trends
        st.markdown("**Quarterly Revenue Trends**")

        fig = go.Figure()
        colors_map = {'OHI': '#00D9FF', 'CTRE': '#FF6B9D', 'SBRA': '#00FF88'}

        for reit_name, rev_df in [('OHI', ohi_rev), ('CTRE', ctre_rev), ('SBRA', sbra_rev)]:
            quarterly = rev_df[rev_df['fp'].str.startswith('Q')].copy()
            quarterly = quarterly.sort_values('end')

            fig.add_trace(go.Scatter(
                x=quarterly['end'],
                y=quarterly['val'] / 1e6,
                mode='lines+markers',
                name=reit_name,
                line=dict(width=3, color=colors_map[reit_name]),
                marker=dict(size=8, color=colors_map[reit_name])
            ))

        fig.update_layout(
            xaxis_title='Quarter End Date',
            yaxis_title='Revenue (Millions USD)',
            hovermode='x unified',
            height=500,
            template="plotly_dark",
            plot_bgcolor='#1e2530',
            paper_bgcolor='#1e2530',
            font=dict(color='#ffffff'),
            legend=dict(
                bgcolor='rgba(30, 37, 48, 0.8)',
                bordercolor='#2e3a4a',
                borderwidth=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ====================================================================
        # SECTION 3: PROFITABILITY
        # ====================================================================
        st.subheader("3. Profitability & Returns")

        # Calculate profitability metrics
        profit_metrics = []

        for reit_name, rev_df, ni_df, equity_df, assets_df in [
            ('OHI', ohi_rev, ohi_ni, ohi_equity, ohi_assets),
            ('CTRE', ctre_rev, ctre_ni, ctre_equity, ctre_assets),
            ('SBRA', sbra_rev, sbra_ni, sbra_equity, sbra_assets)
        ]:
            # TTM revenue
            quarterly_rev = rev_df[rev_df['fp'].str.startswith('Q')].copy()
            quarterly_rev = quarterly_rev.sort_values('end', ascending=False)
            ttm_revenue = quarterly_rev.iloc[0:4]['val'].sum() if len(quarterly_rev) >= 4 else None

            # TTM net income
            quarterly_ni = ni_df[ni_df['fp'].str.startswith('Q')].copy()
            quarterly_ni = quarterly_ni.sort_values('end', ascending=False)
            ttm_ni = quarterly_ni.iloc[0:4]['val'].sum() if len(quarterly_ni) >= 4 else None

            # Latest equity and assets
            latest_equity = equity_df.iloc[0]['val'] if len(equity_df) > 0 else None
            latest_assets = assets_df.iloc[0]['val'] if len(assets_df) > 0 else None

            # Calculate metrics
            net_margin = (ttm_ni / ttm_revenue * 100) if ttm_revenue and ttm_ni else 0
            roe = (ttm_ni / latest_equity * 100) if latest_equity and ttm_ni else 0
            roa = (ttm_ni / latest_assets * 100) if latest_assets and ttm_ni else 0

            profit_metrics.append({
                'REIT': reit_name,
                'TTM Net Income ($M)': ttm_ni / 1e6 if ttm_ni else 0,
                'Net Margin (%)': net_margin,
                'ROE (%)': roe,
                'ROA (%)': roa
            })

        profit_df = pd.DataFrame(profit_metrics)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            highest_margin = profit_df.loc[profit_df['Net Margin (%)'].idxmax(), 'REIT']
            st.metric("Highest Net Margin", highest_margin,
                     f"{profit_df['Net Margin (%)'].max():.2f}%")

        with col2:
            highest_roe = profit_df.loc[profit_df['ROE (%)'].idxmax(), 'REIT']
            st.metric("Highest ROE", highest_roe,
                     f"{profit_df['ROE (%)'].max():.2f}%")

        with col3:
            highest_roa = profit_df.loc[profit_df['ROA (%)'].idxmax(), 'REIT']
            st.metric("Highest ROA", highest_roa,
                     f"{profit_df['ROA (%)'].max():.2f}%")

        with col4:
            total_ni = profit_df['TTM Net Income ($M)'].sum()
            st.metric("Combined Net Income", f"${total_ni:.1f}M", "All 3 REITs")

        # Visualize profitability
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Net Margin (%)', 'Return on Equity (%)',
                          'Return on Assets (%)', 'TTM Net Income ($M)')
        )

        fig.add_trace(go.Bar(x=profit_df['REIT'], y=profit_df['Net Margin (%)'],
                            marker_color='#00D9FF', showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=profit_df['REIT'], y=profit_df['ROE (%)'],
                            marker_color='#00FF88', showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=profit_df['REIT'], y=profit_df['ROA (%)'],
                            marker_color='#FF6B9D', showlegend=False), row=2, col=1)
        fig.add_trace(go.Bar(x=profit_df['REIT'], y=profit_df['TTM Net Income ($M)'],
                            marker_color='#FFA500', showlegend=False), row=2, col=2)

        fig.update_layout(
            height=600,
            title_text="Profitability Metrics",
            template="plotly_dark",
            plot_bgcolor='#1e2530',
            paper_bgcolor='#1e2530',
            font=dict(color='#ffffff')
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ====================================================================
        # SECTION 4: BALANCE SHEET
        # ====================================================================
        st.subheader("4. Balance Sheet Strength & Leverage")

        # Calculate balance sheet metrics
        balance_metrics = []

        for reit_name, assets_df, liab_df, equity_df, debt_df in [
            ('OHI', ohi_assets, ohi_liab, ohi_equity, ohi_debt),
            ('CTRE', ctre_assets, ctre_liab, ctre_equity, ctre_debt),
            ('SBRA', sbra_assets, sbra_liab, sbra_equity, sbra_debt)
        ]:
            latest_assets = assets_df.iloc[0]['val'] if len(assets_df) > 0 else None
            latest_liab = liab_df.iloc[0]['val'] if len(liab_df) > 0 else None
            latest_equity = equity_df.iloc[0]['val'] if len(equity_df) > 0 else None
            latest_debt = debt_df.iloc[0]['val'] if len(debt_df) > 0 else None

            # Calculate ratios
            debt_to_equity = (latest_debt / latest_equity) if latest_debt and latest_equity else 0
            debt_to_assets = (latest_debt / latest_assets * 100) if latest_debt and latest_assets else 0
            equity_to_assets = (latest_equity / latest_assets * 100) if latest_equity and latest_assets else 0

            balance_metrics.append({
                'REIT': reit_name,
                'Total Assets ($B)': latest_assets / 1e9 if latest_assets else 0,
                'Total Debt ($B)': latest_debt / 1e9 if latest_debt else 0,
                'Stockholders Equity ($B)': latest_equity / 1e9 if latest_equity else 0,
                'Debt/Equity Ratio': debt_to_equity,
                'Debt/Assets (%)': debt_to_assets,
                'Equity/Assets (%)': equity_to_assets
            })

        balance_df = pd.DataFrame(balance_metrics)

        # Display metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            lowest_leverage = balance_df.loc[balance_df['Debt/Equity Ratio'].idxmin(), 'REIT']
            st.metric("Lowest Leverage (D/E)", lowest_leverage,
                     f"{balance_df['Debt/Equity Ratio'].min():.2f}x")

        with col2:
            highest_equity = balance_df.loc[balance_df['Stockholders Equity ($B)'].idxmax(), 'REIT']
            st.metric("Strongest Equity", highest_equity,
                     f"${balance_df['Stockholders Equity ($B)'].max():.2f}B")

        with col3:
            avg_leverage = balance_df['Debt/Equity Ratio'].mean()
            st.metric("Average D/E Ratio", f"{avg_leverage:.2f}x", "All 3 REITs")

        # Visualize balance sheet
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Capital Structure', 'Debt/Equity Ratio', 'Leverage (Debt/Assets %)')
        )

        # Stacked bar for capital structure
        fig.add_trace(go.Bar(x=balance_df['REIT'], y=balance_df['Total Debt ($B)'],
                            name='Debt', marker_color='#FF6B9D'), row=1, col=1)
        fig.add_trace(go.Bar(x=balance_df['REIT'], y=balance_df['Stockholders Equity ($B)'],
                            name='Equity', marker_color='#00FF88'), row=1, col=1)

        fig.add_trace(go.Bar(x=balance_df['REIT'], y=balance_df['Debt/Equity Ratio'],
                            marker_color='#FFA500', showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=balance_df['REIT'], y=balance_df['Debt/Assets (%)'],
                            marker_color='#00D9FF', showlegend=False), row=1, col=3)

        fig.update_layout(
            height=400,
            title_text="Balance Sheet Analysis",
            barmode='stack',
            template="plotly_dark",
            plot_bgcolor='#1e2530',
            paper_bgcolor='#1e2530',
            font=dict(color='#ffffff')
        )
        fig.update_yaxes(title_text="Billions USD", row=1, col=1)
        fig.update_yaxes(title_text="Ratio", row=1, col=2)
        fig.update_yaxes(title_text="%", row=1, col=3)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ====================================================================
        # SECTION 5: COMPREHENSIVE SCORECARD
        # ====================================================================
        st.subheader("5. Comprehensive Scorecard")

        # Create comprehensive comparison
        scorecard = pd.DataFrame({
            'Metric': [
                'Total Assets ($B)',
                'Real Estate ($B)',
                'TTM Revenue ($M)',
                'YoY Revenue Growth (%)',
                '3Y Revenue CAGR (%)',
                'Revenue Volatility (%) ↓',
                'TTM Net Income ($M)',
                'Net Margin (%)',
                'ROE (%)',
                'ROA (%)',
                'Stockholders Equity ($B)',
                'Long-term Debt ($B)',
                'Debt/Equity Ratio ↓',
                'Debt/Assets (%) ↓'
            ]
        })

        # Populate with data
        for reit in ['OHI', 'CTRE', 'SBRA']:
            scorecard[reit] = [
                f"{size_df[size_df['REIT']==reit]['Total Assets ($B)'].values[0]:.2f}",
                f"{size_df[size_df['REIT']==reit]['Real Estate ($B)'].values[0]:.2f}",
                f"{size_df[size_df['REIT']==reit]['TTM Revenue ($M)'].values[0]:.1f}",
                f"{growth_df[growth_df['REIT']==reit]['YoY Growth (%)'].values[0]:.2f}",
                f"{growth_df[growth_df['REIT']==reit]['3Y CAGR (%)'].values[0]:.2f}",
                f"{growth_df[growth_df['REIT']==reit]['Revenue Volatility (%)'].values[0]:.2f}",
                f"{profit_df[profit_df['REIT']==reit]['TTM Net Income ($M)'].values[0]:.1f}",
                f"{profit_df[profit_df['REIT']==reit]['Net Margin (%)'].values[0]:.2f}",
                f"{profit_df[profit_df['REIT']==reit]['ROE (%)'].values[0]:.2f}",
                f"{profit_df[profit_df['REIT']==reit]['ROA (%)'].values[0]:.2f}",
                f"{balance_df[balance_df['REIT']==reit]['Stockholders Equity ($B)'].values[0]:.2f}",
                f"{balance_df[balance_df['REIT']==reit]['Total Debt ($B)'].values[0]:.2f}",
                f"{balance_df[balance_df['REIT']==reit]['Debt/Equity Ratio'].values[0]:.2f}",
                f"{balance_df[balance_df['REIT']==reit]['Debt/Assets (%)'].values[0]:.1f}",
            ]

        st.markdown("**↓ = Lower is Better**")
        st.dataframe(scorecard, use_container_width=True)

        # Key findings
        st.markdown("---")
        st.subheader("Key Findings")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**OHI (Omega Healthcare)**")
            st.success("""
            - Largest by assets and revenue
            - Market leader position
            - Stable, mature cash flows
            - Lower growth rate
            """)

        with col2:
            st.markdown("**CTRE (CareTrust)**")
            st.info("""
            - Highest growth profile
            - Strong profitability
            - Nimble and efficient
            - Smallest scale
            """)

        with col3:
            st.markdown("**SBRA (Sabra Healthcare)**")
            st.warning("""
            - Balanced size and growth
            - Diversified approach
            - Moderate risk profile
            - Middle-market position
            """)

    else:
        st.error("Unable to load complete REIT data. Please ensure all data files are available.")

# ============================================================================
# PAGE 6: BLS ECONOMIC DATA
# ============================================================================
elif page == "BLS Economic Data":
    st.header("BLS Economic Indicators - Nursing Care Facilities")

    # Load BLS data
    ppi_data = load_data("bls_ppi_latest.csv")
    employment_data = load_data("bls_employment_latest.csv")

    if ppi_data is None and employment_data is None:
        st.error("BLS data not available. Run `make data-bls` to fetch data.")
        st.info("**Need a BLS API key?** Register for free at: https://data.bls.gov/registrationEngine/")
        st.code("# After getting your API key, add it to .env:\nBLS_API_KEY=your_key_here\n\n# Then run:\nmake data-bls", language="bash")
    else:
        # PPI Analysis
        if ppi_data is not None:
            st.subheader("Producer Price Index (PPI) - NAICS 623110")
            st.markdown("*Nursing Care Facilities (Skilled Nursing Facilities)*")

            # Latest PPI values
            col1, col2, col3, col4 = st.columns(4)

            series_names = ["PPI Overall", "PPI Medicare", "PPI Medicaid", "PPI Private Insurance"]
            cols = [col1, col2, col3, col4]

            for series_name, col in zip(series_names, cols):
                series_data = ppi_data[ppi_data['series_name'] == series_name]
                if len(series_data) > 0:
                    latest = series_data.iloc[0]
                    with col:
                        st.metric(
                            series_name.replace("PPI ", ""),
                            f"{float(latest['value']):.1f}",
                            f"{latest['year']}-{latest['period']}"
                        )

            st.markdown("---")

            # PPI Trend Chart
            st.subheader("PPI Trends Over Time")

            fig = go.Figure()

            colors = {
                "PPI Overall": "#1f77b4",
                "PPI Medicare": "#ff7f0e",
                "PPI Medicaid": "#2ca02c",
                "PPI Private Insurance": "#d62728"
            }

            for series_name in series_names:
                series_data = ppi_data[ppi_data['series_name'] == series_name].copy()
                if len(series_data) > 0:
                    # Create date from year and period
                    series_data = series_data.sort_values(['year', 'period'])

                    fig.add_trace(go.Scatter(
                        x=series_data['year'].astype(str) + "-" + series_data['period'],
                        y=series_data['value'].astype(float),
                        mode='lines+markers',
                        name=series_name,
                        line=dict(width=3, color=colors.get(series_name)),
                        marker=dict(size=6)
                    ))

            fig.update_layout(
                xaxis_title="Period",
                yaxis_title="PPI Index Value",
                hovermode='x unified',
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # PPI Comparison by Payer Type
            st.subheader("Latest PPI by Payer Type")

            latest_ppi = []
            for series_name in series_names[1:]:  # Skip overall
                series_data = ppi_data[ppi_data['series_name'] == series_name]
                if len(series_data) > 0:
                    latest = series_data.iloc[0]
                    payer = series_name.replace("PPI ", "")
                    latest_ppi.append({
                        'Payer Type': payer,
                        'PPI': float(latest['value'])
                    })

            if latest_ppi:
                ppi_df = pd.DataFrame(latest_ppi)
                fig = px.bar(
                    ppi_df,
                    x='Payer Type',
                    y='PPI',
                    color='Payer Type',
                    title='PPI Comparison by Payer Type (Latest Period)',
                    text='PPI'
                )
                fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

        # Employment Analysis
        if employment_data is not None:
            st.subheader("Employment Metrics - Nursing Care Facilities")

            # Latest employment metrics
            col1, col2, col3, col4 = st.columns(4)

            metrics = [
                ("Total Employees", "thousands", col1),
                ("Avg Hourly Earnings", "$", col2),
                ("Avg Weekly Hours", "hours", col3),
                ("Avg Weekly Earnings", "$", col4)
            ]

            for metric_name, unit, col in metrics:
                series_data = employment_data[employment_data['series_name'] == metric_name]
                if len(series_data) > 0:
                    latest = series_data.iloc[0]
                    value = float(latest['value'])

                    if unit == "$":
                        display_val = f"${value:.2f}"
                    elif unit == "thousands":
                        display_val = f"{value:.0f}K"
                    else:
                        display_val = f"{value:.1f}"

                    with col:
                        st.metric(
                            metric_name.replace("Avg ", ""),
                            display_val,
                            f"{latest['year']}-{latest['period']}"
                        )

            st.markdown("---")

            # Employment Trends
            st.subheader("Employment Trends Over Time")

            # Create 2x2 subplot
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Total Employees (thousands)",
                    "Avg Hourly Earnings ($)",
                    "Avg Weekly Hours",
                    "Avg Weekly Earnings ($)"
                )
            )

            employment_metrics = [
                ("Total Employees", 1, 1),
                ("Avg Hourly Earnings", 1, 2),
                ("Avg Weekly Hours", 2, 1),
                ("Avg Weekly Earnings", 2, 2)
            ]

            for metric_name, row, col in employment_metrics:
                series_data = employment_data[employment_data['series_name'] == metric_name].copy()
                if len(series_data) > 0:
                    series_data = series_data.sort_values(['year', 'period'])

                    fig.add_trace(
                        go.Scatter(
                            x=series_data['year'].astype(str) + "-" + series_data['period'],
                            y=series_data['value'].astype(float),
                            mode='lines+markers',
                            name=metric_name,
                            showlegend=False,
                            line=dict(width=2),
                            marker=dict(size=4)
                        ),
                        row=row,
                        col=col
                    )

            fig.update_layout(height=700, showlegend=False)
            fig.update_xaxes(title_text="Period")

            st.plotly_chart(fig, use_container_width=True)

            # Data table
            st.subheader("Latest Employment Data")
            if employment_data is not None:
                latest_employment = employment_data.groupby('series_name').first().reset_index()
                display_cols = ['series_name', 'year', 'period', 'value']
                st.dataframe(
                    latest_employment[display_cols].rename(columns={
                        'series_name': 'Metric',
                        'year': 'Year',
                        'period': 'Period',
                        'value': 'Value'
                    }),
                    use_container_width=True
                )

        # Insights
        st.markdown("---")
        st.subheader("Key Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            **PPI Insights:**
            - Tracks pricing power for nursing facilities
            - Medicare/Medicaid PPI shows reimbursement rate trends
            - Private insurance PPI indicates market pricing
            - Higher PPI = better pricing environment for REITs
            """)

        with col2:
            st.info("""
            **Employment Insights:**
            - Labor is largest cost for nursing facilities
            - Rising wages impact REIT operating margins
            - Staffing levels affect quality ratings
            - Employment trends indicate industry health
            """)

# ============================================================================
# PAGE: ACTUAL REIT PROPERTIES
# ============================================================================
if page == "Actual REIT Properties":
    # Import and run the actual REIT portfolios page
    try:
        from pages.actual_reit_portfolios import main as actual_portfolios_main
        actual_portfolios_main()
    except ImportError as e:
        st.error(f"Error loading Actual REIT Properties page: {e}")
        st.info("Make sure the actual_reit_portfolios module is installed correctly.")

# ============================================================================
# PAGE: DATA EXPLORER
# ============================================================================
elif page == "Data Explorer":
    st.header("Data Explorer")

    st.markdown("Explore all available datasets")

    data_type = st.selectbox(
        "Select Dataset:",
        ["CMS Provider Info", "BLS PPI Data", "BLS Employment Data", "REIT Revenue", "REIT Assets", "REIT Liabilities", "REIT Net Income"]
    )

    if data_type == "CMS Provider Info":
        data = load_data("cms_provider_info_latest.csv")
        st.subheader("CMS Nursing Home Provider Information")
    elif data_type == "BLS PPI Data":
        data = load_data("bls_ppi_latest.csv")
        st.subheader("BLS Producer Price Index Data")
    elif data_type == "BLS Employment Data":
        data = load_data("bls_employment_latest.csv")
        st.subheader("BLS Employment Data")
    elif data_type == "REIT Revenue":
        reit = st.selectbox("Select REIT:", ["OHI", "CTRE", "SBRA"])
        data = load_data(f"sec_{reit.lower()}_revenues_latest.csv")
        st.subheader(f"{reit} Revenue Data")
    elif data_type == "REIT Assets":
        reit = st.selectbox("Select REIT:", ["OHI", "CTRE", "SBRA"])
        data = load_data(f"sec_{reit.lower()}_assets_latest.csv")
        st.subheader(f"{reit} Assets Data")
    elif data_type == "REIT Liabilities":
        reit = st.selectbox("Select REIT:", ["OHI", "CTRE", "SBRA"])
        data = load_data(f"sec_{reit.lower()}_liabilities_latest.csv")
        st.subheader(f"{reit} Liabilities Data")
    elif data_type == "REIT Net Income":
        reit = st.selectbox("Select REIT:", ["OHI", "CTRE", "SBRA"])
        data = load_data(f"sec_{reit.lower()}_netincomeloss_latest.csv")
        st.subheader(f"{reit} Net Income Data")

    if data is not None:
        # Stats
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Columns", len(data.columns))
        with col3:
            if 'end' in data.columns:
                st.metric("Date Range", f"{data['end'].min()} to {data['end'].max()}")

        st.markdown("---")

        # Display data
        st.subheader("Data Preview")
        st.dataframe(data, use_container_width=True)

        # Download button
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{data_type.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    else:
        st.error("Data not available")

# Footer
st.markdown("---")
st.markdown("*Data sources: CMS Provider Data Catalog, SEC EDGAR*")
st.markdown("*Dashboard built with Streamlit • Data updated: October 2025*")
