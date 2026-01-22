# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 14:57:28 2026

@author: Frei.Alexander.P
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import polars as pl
from datetime import datetime

# ---------- Modern Card Style ----------
def inject_styles():
    st.markdown("""
    <style>
    /* --- Card backgrounds and spacing --- */
    .my-card, .stPlotlyChart, .stChart {
      background: #f8faff;
      border-radius: 24px;
      box-shadow: 0 2px 16px rgba(191,205,255,0.08);
      border: 1px solid #dde5fd;
      padding: 8px 0px 13px 20px;
      margin-bottom: 1.5rem;
      
    }
    .stMarkdown h3, .stMarkdown h2, .stMarkdown h1 {
      margin-bottom: 0.32rem !important;
      margin-top: 0.7rem !important;
      color: #1a2337;
    }
    .stMarkdown {
      margin-bottom: 0.7rem !important;
    }
    
    .card-caption {
      font-size: 1.02rem;
      color: #647398;
      margin-bottom: 1.2rem;
      margin-top: -0.3rem;
    }
    /* --- Custom radio styling (red accent) --- */
    label[data-baseweb="radio"] .stRadio label, .stRadio > div, .stRadio div[role="radiogroup"] label {
      color: #213052;
      font-weight: 600;
    }
    div[role="radiogroup"]>div {
      gap: 1.3rem !important;
    }
    /* --- Custom toggle switch --- */
    .stToggleSwitch {
      --toggle-bg-on: #ff3e3e;
      --toggle-bg-off: #e3e8f8;
      --toggle-knob: #fff;
      --toggle-border: #ff3e3e;
      background: var(--toggle-bg-off);
      border-radius: 24px;
      border: 1.5px solid var(--toggle-border);
      transition: background-color 0.14s;
    }
    input:checked + div .stToggle__slider {
      background: var(--toggle-bg-on);
      border-color: var(--toggle-bg-on);
    }
    /* --- Select boxes --- */
    .stSelectbox label, .stSelectbox div[data-baseweb="select"] {
      font-weight: 600;
      color: #465482;
      padding-bottom: 2.5px;
    }
    /* --- Make selectboxes hover snazzy --- */
    .stSelectbox > div[data-baseweb="select"]:hover {
      background:#eef1fa;
      border-color:#ff3e3e !important;
    }
    /* --- Sliders --- */
    .stSlider {
      margin-top: 15px;
    }
    .stSlider .stSlider {
      accent-color: #ff3e3e !important;
    }
    .stSlider label {
      color: #465482;
      font-weight: 600;
    }
    .stSlider span[data-baseweb="slider"] {
      background: #f7f8fc !important;
      border-radius: 10px;
      border: 1.5px solid #dbe2ee;
      padding: 1px 14px;
    }
    /* --- Remove extra padding/gaps --- */
    [data-testid="stVerticalBlock"] {
      padding-top: 0.1rem !important;
      padding-bottom: 0.1rem !important;
    }
    [data-testid="column"] { padding-top: 1px!important;}
    </style>
    """, unsafe_allow_html=True)

inject_styles()
st.set_page_config(page_title="Current Employment Statistics Multi-Series Plotter", layout="wide", page_icon="📈")

# =================================
#            DATA LOAD
# =================================



@st.cache_resource
def load_panel_data_and_maps():
    df_CES = pl.read_parquet("CES_2015_2026_01_14.parquet")
    df_CEU = pl.read_parquet("CEU_2015_2026_01_14.parquet")
    return df_CES, df_CEU

df_CES, df_CEU = load_panel_data_and_maps()

# =================================
#          UTILITY FUNCTIONS
# =================================



def extract_metric_name(series_title, sector=None, industry=None):
    if not series_title or (isinstance(series_title, float) and pl.is_nan(series_title)): return ""
    parts = [p.strip() for p in str(series_title).split(',')]
    if parts and parts[-1].lower() in ["seasonally adjusted", "not seasonally adjusted"]: parts = parts[:-1]
    if len(parts) > 1 and (
        (sector and parts[-1].lower() == sector.lower())
        or (industry and parts[-1].lower() == industry.lower())
    ):
        parts = parts[:-1]
    return ', '.join(parts)

def pretty_industry(sector, industry):
    return f"{industry} (All of sector)" if industry and sector == industry else str(industry) if industry else ""

def get_metrics_for_combo(df, sector, industry):
    if not sector or not industry: return set(), {}
    mask = (pl.col("Commerce_Sector").str.strip_chars().str.to_lowercase() == sector.lower().strip()) & \
           (pl.col("Commerce_Industry").str.strip_chars().str.to_lowercase() == industry.lower().strip())
    subset = df.filter(mask)
    titles = subset["Series_Title"].unique()
    mapping = {}
    for t in titles:
        label = extract_metric_name(t, sector, industry)
        if label: mapping[label] = t
    return set(mapping.keys()), mapping

def intersection_maps(metric_maps):
    metric_maps = [m for m in metric_maps if m]
    if not metric_maps: return {}
    shared = set(metric_maps[0].keys())
    for m in metric_maps[1:]: shared &= set(m.keys())
    return {s: next(m[s] for m in metric_maps if s in m) for s in shared}

def sector_industry_cache(df):
    sector_choices = sorted(df["Commerce_Sector"].unique().to_list(), key=lambda s: (s != "Total nonfarm", s))
    sector_industry_map = {s: df.filter(pl.col("Commerce_Sector") == s)["Commerce_Industry"].unique().to_list() for s in sector_choices}
    pretty_names_map = {(sector, ind): pretty_industry(sector, ind) for sector in sector_choices for ind in sector_industry_map[sector]}
    return sector_choices, sector_industry_map, pretty_names_map

def initialize_session_cache():
    if "sector_choices" not in st.session_state:
        s, si, pn = sector_industry_cache(df_CES)
        st.session_state.sector_choices = s
        st.session_state.sector_industry_map = si
        st.session_state.pretty_names_map = pn

    if "chart3_sectors" not in st.session_state or not st.session_state.chart3_sectors:
        defaults = ["Manufacturing", "Construction", "Mining and logging", "Transportation and warehousing", "Utilities"]
        st.session_state.chart3_sectors = []
        for s in defaults:
            inds_sorted = [s] + sorted([ind for ind in st.session_state.sector_industry_map[s] if ind != s])
            st.session_state.chart3_sectors.append({"sector": s, "industries": [pretty_industry(s, s)]})

    if "chart3_last_sectors" not in st.session_state or not st.session_state.chart3_last_sectors:
        st.session_state.chart3_last_sectors = [d["sector"] for d in st.session_state.chart3_sectors]

initialize_session_cache()

# =================================
#          CHART 1: Overview
# =================================
def chart1_controls(df):
    style_opts = {"index": 0, "horizontal": True}
    sectors = sorted(df["Commerce_Sector"].unique().to_list(), key=lambda s: (s != "Total nonfarm", s))
    sector = st.selectbox("Select sector:", sectors, index=0, key="sect1")
    inds = df.filter(pl.col("Commerce_Sector") == sector)["Commerce_Industry"].unique().to_list()
    inds_sorted = [sector] + sorted([ind for ind in inds if ind != sector])
    inds_pretty = [pretty_industry(sector, ind) for ind in inds_sorted]
    ind_map = dict(zip(inds_pretty, inds_sorted))
    industry_pretty = st.selectbox("Select industry:", inds_pretty, index=0, key="ind1")
    industry = ind_map[industry_pretty]
    metrics, mapping = get_metrics_for_combo(df, sector, industry)
    main_metric = [m for m in metrics if "all employees, thousands" in m.lower()]
    metrics_sorted = main_metric + sorted([m for m in metrics if m not in main_metric])
    metric_default_value = (
        st.session_state.get("ser1", None) if st.session_state.get("ser1", None) in metrics_sorted
        else next((m for m in metrics_sorted if m.lower().startswith("all employees, thousands")), metrics_sorted[0] if metrics_sorted else None)
    )
    metric_choice = st.selectbox("Select metric to plot:", metrics_sorted, index=metrics_sorted.index(metric_default_value) if metric_default_value in metrics_sorted else 0, key="ser1") if metrics_sorted else None
    return sector, industry, inds_pretty, metrics_sorted, metric_choice, mapping

def chart1_plot(df, sector, industry, series_title, value_mode, year_range, metric_label):
    plot_df = df.filter(
        (pl.col("Commerce_Sector").str.strip_chars().str.to_lowercase() == sector.lower().strip()) &
        (pl.col("Commerce_Industry").str.strip_chars().str.to_lowercase() == industry.lower().strip()) &
        (pl.col("Series_Title") == series_title) &
        (pl.col("Date") >= pl.datetime(year_range[0], 1, 1)) &
        (pl.col("Date") <= pl.datetime(year_range[1], 12, 31))
    ).sort("Date")
    if not value_mode and plot_df.height > 0:
        plot_df = plot_df.with_columns(Value=pl.col("Value").diff()).drop_nulls("Value")
    if plot_df.height == 0:
        st.info("Select an available combination and metric to begin.")
        return
    plot_pd = plot_df.with_columns([pl.lit(pretty_industry(sector, industry)).alias("Legend")]).to_pandas()
    value_label = f"{metric_label}{'' if value_mode else ' (MoM Change)'}"
    
    # Extract the Series ID from the first row (it is the same for the entire filtered set)
    series_id = ""
    if plot_df.height > 0 and "Series_ID" in plot_df.columns:
        series_id = plot_df[0, "Series_ID"]
    
    if value_mode:
        fig = px.line(plot_pd, x='Date', y='Value', color='Legend', labels={'Value': value_label, 'Date': '', 'Legend': ''})
        fig.update_traces(line_width=2)
    else:
        fig = px.bar(plot_pd, x='Date', y='Value', color='Legend', barmode="group", labels={'Value': value_label, 'Date': '', 'Legend': ''})

    # Styled, information-rich title
    seasonality_label_1 = "Seasonally Adjusted" if seasonally_adjusted_1 else "Not Seasonally Adjusted"
    fig.update_layout(
        font=dict(color="black"),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, 
            font=dict(color="black"), title=""
        ),
        xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
        yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=520,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=120, b=60),  # <-- key line: L/R/T/B pixels
        title=(
            "<span style='font-size:20px; font-weight:700; color:#23263c;'>Current Employment Statistics</span><br>"
            f"<span style='font-size:15px; color:#4a5a6a;'>Series:</span> "
            f"<span style='font-size:15px; font-weight:600; color:#243350;'>{pretty_industry(sector, industry)}</span>, "
            f"<span style='font-size:14px; color:#6d7787;'>Series ID: {series_id}</span><br>"
            
            f"<span style='font-size:15px; color:#4a5a6a;'>Measure:</span> "
            f"<span style='font-size:15px; font-weight:600; color:#e74a3b;'>{metric_label}{'' if value_mode else ' (MoM Change)'}</span> "
            f"<span style='font-size:14px; color:#243350;'>| {seasonality_label_1}</span>"
        )
    )

    for trace in fig.data:
        trace.hovertemplate = f'<b>{trace.name}</b><br>%{{x|%b %Y}}: %{{y:,.0f}}<extra></extra>'
    st.plotly_chart(fig, use_container_width=True)

# =================================
#      CHART 2: Flexible Compare
# =================================
def chart2_controls(df):
    SECTOR_CHOICES = 3
    all_sectors = sorted(df["Commerce_Sector"].unique().to_list(), key=lambda s: (s != "Total nonfarm", s))
    sector_columns = st.columns(SECTOR_CHOICES, gap="small")
    industry_columns = st.columns(SECTOR_CHOICES, gap="small")
    defaults = [("Total nonfarm", "Total nonfarm"), ("Total private", "Total private"), ("", "")]
    sector_selections, industry_selections, ind_pretty_selections = [], [], []
    for i, col in enumerate(sector_columns):
        default_sector = defaults[i][0]
        sector = col.selectbox(f"Sector {i+1}", options=[""] + all_sectors, index=all_sectors.index(default_sector)+1 if default_sector else 0, key=f"cmp_sect{i}")
        sector_selections.append(sector if sector else None)
    for i, (sect, col) in enumerate(zip(sector_selections, industry_columns)):
        if sect:
            inds = df.filter(pl.col("Commerce_Sector") == sect)["Commerce_Industry"].unique().to_list()
            inds_sorted = [sect] + sorted([ind for ind in inds if ind != sect])
            inds_pretty = [pretty_industry(sect, ind) for ind in inds_sorted]
            pretty = col.selectbox(f"Industry {i+1}", options=[""] + inds_pretty, index=inds_pretty.index(pretty_industry(sect, defaults[i][1]))+1 if defaults[i][1] in inds_sorted else 0, key=f"cmp_ind{i}") if inds_pretty else ""
            ind_map = dict(zip(inds_pretty, inds_sorted))
            ind_sel = ind_map[pretty] if pretty else None
            industry_selections.append(ind_sel)
            ind_pretty_selections.append(pretty)
        else: industry_selections.append(None); ind_pretty_selections.append("")
    chosen_combos = [(sector_selections[i], industry_selections[i], ind_pretty_selections[i]) for i in range(SECTOR_CHOICES) if sector_selections[i] and industry_selections[i]]
    metric_maps = [get_metrics_for_combo(df, sect, ind)[1] for (sect, ind, _) in chosen_combos] if chosen_combos else []
    common_metrics_map = intersection_maps(metric_maps) if metric_maps else {}
    main_metrics = [m for m in common_metrics_map.keys() if "all employees, thousands" in m.lower()]
    metrics_sorted = main_metrics + sorted([m for m in common_metrics_map.keys() if m not in main_metrics])
    return chosen_combos, metrics_sorted, common_metrics_map

def chart2_plot(df, chosen_combos, metric_choice, value_mode, year_range):
    frames = []
    for (sect, ind, ind_pretty) in chosen_combos:
        _, metric_map = get_metrics_for_combo(df, sect, ind)
        if metric_choice and metric_choice in metric_map:
            fulltitle = metric_map[metric_choice]
            tmp = df.filter(
                (pl.col("Commerce_Sector").str.strip_chars().str.to_lowercase() == sect.lower().strip()) &
                (pl.col("Commerce_Industry").str.strip_chars().str.to_lowercase() == ind.lower().strip()) &
                (pl.col("Series_Title") == fulltitle) &
                (pl.col("Date") >= pl.datetime(year_range[0], 1, 1)) &
                (pl.col("Date") <= pl.datetime(year_range[1], 12, 31))
            ).sort("Date")
            if not value_mode and tmp.height > 0:
                tmp = tmp.with_columns(Value=pl.col("Value").diff()).drop_nulls("Value")
            if tmp.height > 0:
                tmp = tmp.with_columns([pl.lit(pretty_industry(sect, ind)).alias("Legend")])
                frames.append(tmp)
    plot_df = pl.concat(frames) if frames else pl.DataFrame()
    if plot_df.height == 0 or not metric_choice:
        st.warning("No data available for the combination selected.")
        return
    plot_pd = plot_df.to_pandas()
    color_palette = px.colors.qualitative.Set2 + px.colors.qualitative.Plotly + px.colors.qualitative.D3
    unique_legends = plot_pd["Legend"].unique()
    color_discrete_map = dict(zip(unique_legends, color_palette * ((len(unique_legends)+len(color_palette)-1)//len(color_palette))))
    value_label_2 = f"{metric_choice}{'' if value_mode else ' (MoM Change)'}"

    if value_mode:
        fig = px.line(plot_pd, x='Date', y='Value', color='Legend', color_discrete_map=color_discrete_map, labels={'Value': value_label_2, 'Legend': '', 'Date': ''})
        fig.update_traces(line_width=2)
    else:
        fig = px.bar(plot_pd, x='Date', y='Value', color='Legend', barmode="group", color_discrete_map=color_discrete_map, labels={'Value': value_label_2, 'Legend': '', 'Date': ''})
    seasonality_label_1 = "Seasonally Adjusted" if seasonally_adjusted_1 else "Not Seasonally Adjusted"
    dynamic_title = " — ".join([pretty_industry(sect, ind) for (sect, ind, _) in chosen_combos])

    fig.update_layout(
        font=dict(color="black"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(color="black"), title=""),
        xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
        yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=520,
        margin=dict(l=50, r=50, t=120, b=60),  # <-- key line: L/R/T/B pixels
        title=(
            f"<span style='font-size:20px; font-weight:700; color:#23263c;'>{dynamic_title}</span><br>"
            f"<span style='font-size:15px; color:#4a5a6a;'>Measure:</span> "
            f"<span style='font-size:15px; font-weight:600; color:#e74a3b;'>{metric_choice}{'' if value_mode else ' (MoM Change)'}</span> "
            f"<span style='font-size:14px; color:#243350;'>| {seasonality_label_1}</span>")
        )
    for trace in fig.data:
        trace.hovertemplate = f'<b>{trace.name}</b><br>%{{x|%b %Y}}: %{{y:,.0f}}<extra></extra>'
    st.plotly_chart(fig, use_container_width=True)

# =================================
#      CHART 3: YoY Stacked Bar
# =================================
def chart3_column_controls():
    remove_sector_key = "remove_sector_idx"
    # Header row, only once
    header_cols = st.columns([2.5, 5, 1], gap="small")
    header_cols[0].markdown("<div style='font-size:13px; color:#465482; margin-bottom:0.18rem;'>Select sector:</div>", unsafe_allow_html=True)
    header_cols[1].markdown("<div style='font-size:13px; color:#465482; margin-bottom:0.18rem;'>Select industry:</div>", unsafe_allow_html=True)
    header_cols[2].markdown(" ")

    for idx, entry in enumerate(st.session_state.chart3_sectors):
        cols = st.columns([2.5, 5, 1], gap="small")
        sector_default = entry["sector"]
        inds_default = entry["industries"]

        # No label above, just the selectbox
        sector = cols[0].selectbox(
            "",
            [""] + st.session_state.sector_choices,
            index=([""] + st.session_state.sector_choices).index(sector_default) if sector_default in [""] + st.session_state.sector_choices else 0,
            key=f"c3_sector_{idx}",
            label_visibility="collapsed",
        )

        if sector:
            inds_sorted = [sector] + sorted([ind for ind in st.session_state.sector_industry_map[sector] if ind != sector])
            inds_pretty = [pretty_industry(sector, ind) for ind in inds_sorted]
            default_inds = st.session_state.get(f"c3_multi_{idx}", None)
            industries_selected = default_inds if default_inds is not None else inds_default
            industries_selected = [ind for ind in industries_selected if ind in inds_pretty]
            industries_selected = cols[1].multiselect(
                "",
                options=inds_pretty,
                default=industries_selected,
                key=f"c3_multi_{idx}",
                label_visibility="collapsed",
            )
        else:
            inds_pretty, industries_selected = [], []
        if cols[2].button("❌", key=f"c3remove_{idx}"):
            st.session_state[remove_sector_key] = idx
            st.rerun()
        st.session_state.chart3_sectors[idx] = {"sector": sector, "industries": industries_selected}

    bcols = st.columns([2, 8])
    if bcols[0].button("➕", help="Add another sector row", key="c3add") and len(st.session_state.chart3_sectors) < 8:
        st.session_state.chart3_sectors.append({"sector": "", "industries": []})
        st.session_state.chart3_last_sectors.append("")
        st.rerun()

def chart3_plot(df, selected, metric_choice, year_range):
    chart_filters = []
    for sect, ind, legend_label in selected:
        _, metric_map = get_metrics_for_combo(df, sect, ind)
        if metric_choice and metric_choice in metric_map:
            metric_title = metric_map[metric_choice]
            chart_filters.append({
                "sector": sect,
                "industry": ind,
                "metric_title": metric_title,
                "legend": legend_label
            })
    chart3_rows = []
    if chart_filters:
        date_start, date_end = pl.datetime(year_range[0], 1, 1), pl.datetime(year_range[1], 12, 31)
        df_filtered = df.filter((pl.col("Date") >= date_start) & (pl.col("Date") <= date_end))
        for f in chart_filters:
            filtered = (
                df_filtered.filter(
                    (pl.col("Commerce_Sector").str.strip_chars().str.to_lowercase() == f["sector"].lower().strip()) &
                    (pl.col("Commerce_Industry").str.strip_chars().str.to_lowercase() == f["industry"].lower().strip()) &
                    (pl.col("Series_Title") == f["metric_title"])
                )
                .sort("Date")
                .with_columns([
                    pl.col("Value").diff(n=12).alias("YoY_Change"),
                    pl.lit(f["legend"]).alias("Legend"),
                    pl.col("Date")
                ])
                .select(["Date", "YoY_Change", "Legend"])
                .drop_nulls("YoY_Change")
            )
            if filtered.height > 0:
                chart3_rows.append(filtered)
    if chart3_rows:
        chart3_df = pl.concat(chart3_rows)
        chart3_pd = chart3_df.to_pandas()
        chart3_pd['Month'] = chart3_pd['Date'].dt.to_period('M').astype(str)
        fig = px.bar(
            chart3_pd, x="Month", y="YoY_Change", color="Legend",
            labels={"YoY_Change": metric_choice + " (YoY Δ)", "Legend": ""}, barmode="relative"
        )
        if not chart3_pd.empty:
            total_line_df = (
                chart3_pd.groupby("Month", as_index=False)["YoY_Change"].sum().rename(columns={"YoY_Change": "Total Employment Growth"})
            )
            fig.add_trace(go.Scatter(
                x=total_line_df["Month"], y=total_line_df["Total Employment Growth"],
                name="Total Employment Growth", mode="lines+markers",
                marker=dict(color="black", size=7), line=dict(color="black", width=3, dash="dash"),
                hovertemplate="<b>Total Employment Growth</b><br>Month: %{x}<br>Growth: %{y:,.0f}<extra></extra>"
            ))

        seasonality_label_1 = "Seasonally Adjusted" if seasonally_adjusted_1 else "Not Seasonally Adjusted"
        fig.update_layout(
            font=dict(color="#23263c"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="center", x=0.5, font=dict(color="#243350"), title=""),
            xaxis=dict(title=""),
            yaxis=dict(title_font=dict(color="#23263c"), tickfont=dict(color="#23263c")),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=520,
            margin=dict(l=18, r=18, t=48, b=46),
            title=(
                f"<span style='font-size:20px; font-weight:700; color:#23263c;'>Year-on-Year Change of US Employment by Sector</span><br>"
                f"<span style='font-size:15px; color:#4a5a6a;'>Measure:</span> "
                f"<span style='font-size:15px; font-weight:600; color:#e74a3b;'>{metric_choice}</span> "
                f"<span style='font-size:14px; color:#243350;'>| {seasonality_label_1}</span>")
            
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add sector/industry columns and select at least one for comparison.")

# =================================
#           PAGE LAYOUT
# =================================
st.title("CES/CEU Multi-Series Plotter")
col1, col2 = st.columns(2, gap="large")

# ---------- Chart 1 ----------
with col1:
    st.markdown("### National-Level CES/CEU Trends by Sector & Industry")
    st.caption("Choose sector, industry, and metric to visualize employment trends below.")
    row_display1, row_season1 = st.columns(2)
    series_type_1 = row_display1.radio(
        "Display as (Chart 1):",
        ["Employment Level (Total)", "Monthly Change"],
        index=0,
        horizontal=True,
        key="series_type_1"
    )
    # --- Toggle replaces radio for seasonality ---
    seasonally_adjusted_1 = row_season1.toggle(
        "Seasonally Adjusted", value=True, key="season1"
    )
    level_mode_1 = (series_type_1 == "Employment Level (Total)")
    df_1 = df_CES if seasonally_adjusted_1 else df_CEU

    sector, industry, inds_pretty, metrics_sorted, metric_choice, mapping = chart1_controls(df_1)
    series_title_1 = mapping[metric_choice] if metric_choice else None

    dates1 = df_1["Date"].sort().unique().to_list()
    min_year1, max_year1 = dates1[0].year, dates1[-1].year if dates1 else (2020, 2026)
    default_slider1 = (max(min_year1, max_year1 - (5 if level_mode_1 else 1)), max_year1)
    year_range1 = st.slider(
        "Select time range:",
        min_value=min_year1,
        max_value=max_year1,
        value=default_slider1,
        label_visibility="collapsed",
        key="slider1"
    )
    if series_title_1:
        chart1_plot(df_1, sector, industry, series_title_1, level_mode_1, year_range1, metric_choice)
        
# --------- Chart 2 ----------
with col2:
    st.markdown("### Flexible Compare: Up to 3 Sectors/Industries")
    st.caption("Compare up to three CES/CEU sector/industry metrics (same time range and metric across all).")
    row_display2, row_season2 = st.columns(2)
    series_type_2 = row_display2.radio(
        "Display as (Compare Chart):",
        ["Employment Level (Total)", "Monthly Change"],
        index=0,
        horizontal=True,
        key="series_type_2"
    )
    # --- Toggle replaces radio for seasonality ---
    seasonally_adjusted_2 = row_season2.toggle(
        "Seasonally Adjusted",
        value=True,
        key="season2"
    )
    level_mode_2 = (series_type_2 == "Employment Level (Total)")
    df_2 = df_CES if seasonally_adjusted_2 else df_CEU

    chosen_combos, metrics_sorted_2, common_metrics_map = chart2_controls(df_2)

    # Metric selector
    if metrics_sorted_2:
        metric_choice_2 = st.selectbox(
            "Select metric to plot (in all series):",
            metrics_sorted_2,
            index=0,
            key="metric_2"
        )
    else:
        metric_choice_2 = None

    dates2 = df_2["Date"].sort().unique().to_list()
    min_year2, max_year2 = dates2[0].year, dates2[-1].year if dates2 else (2020, 2026)
    default_slider2 = (
        max(min_year2, max_year2 - (5 if level_mode_2 else 1)),
        max_year2
    )
    year_range2 = st.slider(
        "Select time range (compare chart):",
        min_value=min_year2,
        max_value=max_year2,
        value=default_slider2,
        label_visibility="collapsed",
        key="slider2"
    )
    if chosen_combos and metric_choice_2:
        chart2_plot(df_2, chosen_combos, metric_choice_2, level_mode_2, year_range2)
    elif not metrics_sorted_2:
        st.info("Select at least one sector/industry pair with a common metric.")
# --------- Chart 3 ----------
st.markdown("<h3>📊 <b>Monthly Year-over-Year Change (Stacked Bar)</b></h3>", unsafe_allow_html=True)
col_left3, col_right3 = st.columns([2.2, 5], gap="large")

with col_left3:
    st.markdown("""<div class='card-caption'>
    <b>➕</b> to add a sector, <b>❌</b> to remove (see below).</span>
    """, unsafe_allow_html=True)
    st.divider()

    # Everything that is now inside your `with col3:` block -- but minus the plot!
    remove_sector_key = "remove_sector_idx"
    if st.session_state.get(remove_sector_key, None) is not None:
        idx_to_remove = st.session_state[remove_sector_key]
        if 0 <= idx_to_remove < len(st.session_state.chart3_sectors) and len(st.session_state.chart3_sectors) > 1:
            del st.session_state.chart3_sectors[idx_to_remove]
            del st.session_state.chart3_last_sectors[idx_to_remove]
            rem_key = f'c3_multi_{idx_to_remove}'
            if rem_key in st.session_state: del st.session_state[rem_key]
        st.session_state[remove_sector_key] = None
        st.rerun()
    chart3_column_controls()

    # All the input and selection logic—including optcols, toggles, slider, and calculating `selected`, `metric_choice`, etc
    selected = []
    legend_map = {}
    for entry in st.session_state.chart3_sectors:
        sector = entry["sector"]
        industries = entry["industries"] if entry["industries"] else []
        if sector and industries:
            inds_sorted = [sector] + sorted([ind for ind in st.session_state.sector_industry_map[sector] if ind != sector])
            inds_pretty = [pretty_industry(sector, ind) for ind in inds_sorted]
            ind_map = dict(zip(inds_pretty, inds_sorted))
            for ind_pretty in industries:
                sec = sector
                ind = ind_map[ind_pretty]
                legend_label = f"{sector} — {ind}"
                selected.append((sec, ind, legend_label))
                legend_map[(sec, ind)] = legend_label
    metric_maps = [get_metrics_for_combo(df_CES, sect, ind)[1] for sect, ind, _ in selected] if selected else []
    common_metrics_map = intersection_maps(metric_maps) if metric_maps else {}
    main_metrics = [m for m in common_metrics_map.keys() if "all employees, thousands" in m.lower()]
    metrics_sorted = main_metrics + sorted([m for m in common_metrics_map.keys() if m not in main_metrics])
    optcols = st.columns(2)
    with optcols[0]:
        if metrics_sorted:
            preferred = "all employees, thousands"
            default_idx = next(
                (i for i, m in enumerate(metrics_sorted) if m.lower().startswith(preferred)),
                0
            )
            metric_choice = st.selectbox(
                "Metric to compare:",
                metrics_sorted,
                index=default_idx,
                key="ct3_metric"
            )
        else:
            metric_choice = None
            st.info("Choose sector(s)/industry(ies) above to see shared metrics.", icon="🔎")
    with optcols[1]:
        seasonally_adjusted_3 = st.toggle(
            "Seasonally Adjusted",
            value=True,
            key="ct3_season"
        )
    df_3 = df_CES if seasonally_adjusted_3 else df_CEU

    if len(df_3) > 0:
        min_year3, max_year3 = df_3["Date"].min().year, df_3["Date"].max().year
        slider_start = max(min_year3, 2022)
        year_range3 = st.slider(
            "Year range:",
            min_value=min_year3,
            max_value=max_year3,
            value=(slider_start, max_year3),
            step=1,
            key="ct3_years",
            label_visibility="collapsed",
        )
    else:
        year_range3 = (2022, 2026)

with col_right3:
    chart3_plot(df_3, selected, metric_choice, year_range3)
    st.divider()

st.markdown("<br><span style='color: gray; font-size: 13px;'>Source: U.S. Bureau of Labor Statistics, CES/CEU (Current Employment Statistics)</span>", unsafe_allow_html=True)


