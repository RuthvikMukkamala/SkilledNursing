"""
Actual REIT Portfolios Dashboard

Geographic visualization and analysis of actual REIT property lists from:
- OHI: Official facilities list from investor relations (Q2 2025)
- SBRA: Official properties list from investor relations (Q2 2025)
- CTRE: Official facilities list from investor relations (Q2 2025)

"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# REIT colors for consistent visualization
REIT_COLORS = {
    'OHI': '#1f77b4',   # Blue
    'CTRE': '#ff7f0e',  # Orange
    'SBRA': '#2ca02c'   # Green
}

@st.cache_data
def load_ohi_properties():
    """Load OHI properties from parsed PDF."""
    path = Path("data/processed/ohi_facilities_q2_2025.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_sbra_properties():
    """Load SBRA properties from Excel."""
    path = Path("data/processed/sbra_properties_06_30_25.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_ctre_properties():
    """Load CTRE properties from processed data."""
    path = Path("data/processed/ctre_properties_06_30_25.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

def get_state_summary(df, reit_name):
    """Create state-level summary for a REIT."""
    # Filter to US only
    if 'country' in df.columns:
        df = df[df['country'] == 'US']
    elif 'COUNTRY' in df.columns:
        df = df[df['COUNTRY'] == 'US']

    if 'STATE' in df.columns:
        state_col = 'STATE'
    else:
        state_col = 'state'

    if 'PROPERTY_TYPE' in df.columns:
        type_col = 'PROPERTY_TYPE'
    else:
        type_col = 'facility_type'

    summary = df.groupby(state_col).agg({
        type_col: 'count'
    }).rename(columns={type_col: 'property_count'})

    summary = summary.reset_index()
    summary = summary.rename(columns={state_col: 'state'})
    summary = summary.sort_values('property_count', ascending=False)

    return summary

def create_choropleth_map(state_summary, reit_name):
    """Create US choropleth map."""
    colorscale_map = {
        'OHI': 'Blues',
        'SBRA': 'Greens',
        'CTRE': 'Purples'
    }

    fig = go.Figure(data=go.Choropleth(
        locations=state_summary['state'],
        z=state_summary['property_count'],
        locationmode='USA-states',
        colorscale=colorscale_map.get(reit_name, 'Blues'),
        colorbar_title='Properties',
        hovertemplate='<b>%{location}</b><br>Properties: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title=f'{reit_name} Property Distribution (Q2 2025)',
        geo_scope='usa',
        height=500,
        font=dict(size=12)
    )

    return fig

def create_state_bar_chart(state_summary, reit_name):
    """Create bar chart of top states."""
    top_15 = state_summary.head(15)

    fig = go.Figure(data=[
        go.Bar(
            x=top_15['state'],
            y=top_15['property_count'],
            marker_color=REIT_COLORS[reit_name],
            text=top_15['property_count'],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title=f'{reit_name} Top 15 States by Property Count',
        xaxis_title='State',
        yaxis_title='Number of Properties',
        height=400,
        showlegend=False
    )

    return fig

def create_property_type_chart(df, reit_name):
    """Create property type breakdown."""
    if 'PROPERTY_TYPE' in df.columns:
        type_col = 'PROPERTY_TYPE'
    else:
        type_col = 'facility_type'

    type_counts = df[type_col].value_counts()

    fig = go.Figure(data=[
        go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            hole=0.3,
            marker=dict(colors=[REIT_COLORS[reit_name], '#aec7e8', '#ffbb78'])
        )
    ])

    fig.update_layout(
        title=f'{reit_name} Property Type Distribution',
        height=400
    )

    return fig

def create_regional_breakdown(df, reit_name):
    """Create regional breakdown chart."""
    # Filter to US
    if 'country' in df.columns:
        df = df[df['country'] == 'US'].copy()

    if 'STATE' in df.columns:
        state_col = 'STATE'
    else:
        state_col = 'state'

    # Regional groupings
    regions = {
        'Northeast': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
        'Midwest': ['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'],
        'South': ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX'],
        'West': ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA']
    }

    def get_region(state):
        for region, states in regions.items():
            if state in states:
                return region
        return 'Other'

    df['region'] = df[state_col].apply(get_region)
    regional_counts = df['region'].value_counts()

    fig = go.Figure(data=[
        go.Bar(
            x=regional_counts.index,
            y=regional_counts.values,
            marker_color=REIT_COLORS[reit_name],
            text=regional_counts.values,
            textposition='outside'
        )
    ])

    fig.update_layout(
        title=f'{reit_name} Regional Distribution',
        xaxis_title='Region',
        yaxis_title='Properties',
        height=400
    )

    return fig

def create_portfolio_comparison(ohi_df, sbra_df, ctre_df):
    """Create side-by-side portfolio comparison."""
    comparison_data = []

    if ohi_df is not None:
        ohi_us = ohi_df[ohi_df['country'] == 'US'] if 'country' in ohi_df.columns else ohi_df
        ohi_uk = ohi_df[ohi_df['country'] == 'UK'] if 'country' in ohi_df.columns else pd.DataFrame()

        comparison_data.append({
            'REIT': 'OHI',
            'Total Properties': len(ohi_df),
            'US Properties': len(ohi_us),
            'UK Properties': len(ohi_uk),
            'US States': ohi_us['state'].nunique(),
            'Skilled Nursing': len(ohi_df[ohi_df['facility_type'] == 'Skilled Nursing Facility']),
            'Assisted Living': len(ohi_df[ohi_df['facility_type'] == 'Assisted Living Facility']),
        })

    if sbra_df is not None:
        sbra_us = sbra_df[sbra_df['COUNTRY'] == 'US'] if 'COUNTRY' in sbra_df.columns else sbra_df

        comparison_data.append({
            'REIT': 'SBRA',
            'Total Properties': len(sbra_df),
            'US Properties': len(sbra_us),
            'UK Properties': 0,
            'US States': sbra_us['STATE'].nunique(),
            'Skilled Nursing': len(sbra_df[sbra_df['PROPERTY_TYPE'] == 'Skilled Nursing/Transitional Care']),
            'Assisted Living': len(sbra_df[sbra_df['PROPERTY_TYPE'].str.contains('Senior Housing', na=False)]),
        })

    if ctre_df is not None:
        ctre_us = ctre_df[ctre_df['COUNTRY'] == 'US'] if 'COUNTRY' in ctre_df.columns else ctre_df
        ctre_uk = ctre_df[ctre_df['COUNTRY'] == 'UK'] if 'COUNTRY' in ctre_df.columns else pd.DataFrame()

        comparison_data.append({
            'REIT': 'CTRE',
            'Total Properties': len(ctre_df),
            'US Properties': len(ctre_us),
            'UK Properties': len(ctre_uk),
            'US States': ctre_us['STATE'].nunique(),
            'Skilled Nursing': len(ctre_df[ctre_df['PROPERTY_TYPE'] == 'SNF']),
            'Assisted Living': len(ctre_df[ctre_df['PROPERTY_TYPE'] == 'AL']),
        })

    return pd.DataFrame(comparison_data)

def main():
    st.title("Actual REIT Property Portfolios")
    st.markdown("### Official property data from REIT investor relations")

    st.info("""
    **Data Sources:**
    - **OHI**: Facilities list from Omega Healthcare investor relations (Q2 2025)
    - **SBRA**: Properties list from Sabra Health Care REIT (Q2 2025)
    - **CTRE**: Facilities list from CareTrust REIT investor relations (Q2 2025)

    """)

    # Load data
    ohi_df = load_ohi_properties()
    sbra_df = load_sbra_properties()
    ctre_df = load_ctre_properties()

    # Portfolio comparison overview
    st.header("Portfolio Overview")

    if ohi_df is not None or sbra_df is not None or ctre_df is not None:
        comparison = create_portfolio_comparison(ohi_df, sbra_df, ctre_df)
        st.dataframe(comparison, use_container_width=True)

        # Visualize comparison
        col1, col2 = st.columns(2)

        with col1:
            fig_total = go.Figure(data=[
                go.Bar(
                    x=comparison['REIT'],
                    y=comparison['Total Properties'],
                    marker_color=[REIT_COLORS[r] for r in comparison['REIT']],
                    text=comparison['Total Properties'],
                    textposition='outside'
                )
            ])
            fig_total.update_layout(
                title='Total Properties',
                yaxis_title='Properties',
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_total, use_container_width=True)

        with col2:
            fig_states = go.Figure(data=[
                go.Bar(
                    x=comparison['REIT'],
                    y=comparison['US States'],
                    marker_color=[REIT_COLORS[r] for r in comparison['REIT']],
                    text=comparison['US States'],
                    textposition='outside'
                )
            ])
            fig_states.update_layout(
                title='US States Coverage',
                yaxis_title='States',
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_states, use_container_width=True)

    # Tabs for individual REITs
    tab1, tab2, tab3 = st.tabs(["OHI Portfolio", "SBRA Portfolio", "CTRE Portfolio"])

    # OHI TAB
    with tab1:
        if ohi_df is not None:
            st.header("Omega Healthcare (OHI) Properties")

            # Key metrics
            ohi_us = ohi_df[ohi_df['country'] == 'US'] if 'country' in ohi_df.columns else ohi_df
            ohi_uk = ohi_df[ohi_df['country'] == 'UK'] if 'country' in ohi_df.columns else pd.DataFrame()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Properties", f"{len(ohi_df):,}")
            col2.metric("US Properties", f"{len(ohi_us):,}")
            col3.metric("UK Properties", f"{len(ohi_uk):,}")
            col4.metric("US States", f"{ohi_us['state'].nunique()}")

            # Geographic analysis
            st.subheader("Geographic Distribution")

            state_summary = get_state_summary(ohi_df, 'OHI')

            col1, col2 = st.columns([2, 1])

            with col1:
                fig_map = create_choropleth_map(state_summary, 'OHI')
                st.plotly_chart(fig_map, use_container_width=True)

            with col2:
                st.markdown("**Top 10 States**")
                st.dataframe(
                    state_summary.head(10)[['state', 'property_count']],
                    use_container_width=True,
                    hide_index=True
                )

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                fig_bar = create_state_bar_chart(state_summary, 'OHI')
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                fig_types = create_property_type_chart(ohi_df, 'OHI')
                st.plotly_chart(fig_types, use_container_width=True)

            # Regional breakdown
            fig_regions = create_regional_breakdown(ohi_df, 'OHI')
            st.plotly_chart(fig_regions, use_container_width=True)

            # Data table
            with st.expander("View Full Property List"):
                st.dataframe(ohi_df, use_container_width=True)

        else:
            st.warning("OHI property data not found. Please run the parser first.")

    # SBRA TAB
    with tab2:
        if sbra_df is not None:
            st.header("Sabra Health Care (SBRA) Properties")

            # Key metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Properties", f"{len(sbra_df):,}")
            col2.metric("US States", f"{sbra_df['STATE'].nunique()}")
            col3.metric("Property Types", f"{sbra_df['PROPERTY_TYPE'].nunique()}")

            # Geographic analysis
            st.subheader("Geographic Distribution")

            state_summary = get_state_summary(sbra_df, 'SBRA')

            col1, col2 = st.columns([2, 1])

            with col1:
                fig_map = create_choropleth_map(state_summary, 'SBRA')
                st.plotly_chart(fig_map, use_container_width=True)

            with col2:
                st.markdown("**Top 10 States**")
                st.dataframe(
                    state_summary.head(10)[['state', 'property_count']],
                    use_container_width=True,
                    hide_index=True
                )

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                fig_bar = create_state_bar_chart(state_summary, 'SBRA')
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                fig_types = create_property_type_chart(sbra_df, 'SBRA')
                st.plotly_chart(fig_types, use_container_width=True)

            # Regional breakdown
            fig_regions = create_regional_breakdown(sbra_df, 'SBRA')
            st.plotly_chart(fig_regions, use_container_width=True)

            # Data table
            with st.expander("View Full Property List"):
                st.dataframe(sbra_df, use_container_width=True)

        else:
            st.warning("SBRA property data not found.")

    # CTRE TAB
    with tab3:
        if ctre_df is not None:
            st.header("CareTrust REIT (CTRE) Properties")

            # Key metrics
            ctre_us = ctre_df[ctre_df['COUNTRY'] == 'US'] if 'COUNTRY' in ctre_df.columns else ctre_df
            ctre_uk = ctre_df[ctre_df['COUNTRY'] == 'UK'] if 'COUNTRY' in ctre_df.columns else pd.DataFrame()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Properties", f"{len(ctre_df):,}")
            col2.metric("US Properties", f"{len(ctre_us):,}")
            col3.metric("UK Properties", f"{len(ctre_uk):,}")
            col4.metric("US States", f"{ctre_us['STATE'].nunique()}")

            # Add bed count if available
            if 'BED_COUNT' in ctre_df.columns:
                col1, col2 = st.columns(2)
                col1.metric("Total Beds", f"{ctre_df['BED_COUNT'].sum():,}")
                col2.metric("Avg Beds per Facility", f"{ctre_df['BED_COUNT'].mean():.0f}")

            # Geographic analysis
            st.subheader("Geographic Distribution")

            state_summary = get_state_summary(ctre_df, 'CTRE')

            col1, col2 = st.columns([2, 1])

            with col1:
                fig_map = create_choropleth_map(state_summary, 'CTRE')
                st.plotly_chart(fig_map, use_container_width=True)

            with col2:
                st.markdown("**Top 10 States**")
                st.dataframe(
                    state_summary.head(10)[['state', 'property_count']],
                    use_container_width=True,
                    hide_index=True
                )

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                fig_bar = create_state_bar_chart(state_summary, 'CTRE')
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                fig_types = create_property_type_chart(ctre_df, 'CTRE')
                st.plotly_chart(fig_types, use_container_width=True)

            # Regional breakdown
            fig_regions = create_regional_breakdown(ctre_df, 'CTRE')
            st.plotly_chart(fig_regions, use_container_width=True)

            # Data table
            with st.expander("View Full Property List"):
                st.dataframe(ctre_df, use_container_width=True)

        else:
            st.warning("CTRE property data not found.")

if __name__ == "__main__":
    main()
