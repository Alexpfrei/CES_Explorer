import streamlit as st
import plotly.express as px
import polars as pl
from datetime import datetime
import plotly.graph_objs as go
# -- Modern styling for compact, card-like layout and grouped controls
st.set_page_config(page_title="CES/CEU Multi-Series Plotter", layout="wide", page_icon="📈")
# ---------- Modern Card Style ----------
st.markdown("""
<style>
[data-testid="stVerticalBlock"] {
  padding-top: 0rem !important; padding-bottom: 0rem !important;
}
.stChart, .stPlotlyChart, .my-card {
  background: linear-gradient(135deg,rgba(240,243,250,0.93) 0%,rgba(230,235,244,0.98) 100%);
  border-radius: 24px;
  box-shadow: 0 3px 20px rgba(140,160,212,0.10),0 1.5px 4px rgba(140,140,140,0.06);
  padding: 18px 16px 7px 16px;
  margin-bottom: 1.25rem;
  border: 1px solid rgba(200,210,226,0.18);
}
.stMarkdown h3, .stMarkdown h2, .stMarkdown h1 {
  margin-bottom: 0.4rem !important;
  margin-top: 0.6rem !important;
}
.stMarkdown {
  margin-bottom: 0.7rem !important;
}
.card-caption {
  font-size: 0.95rem !important;
  color: #4a5c73 !important;
  margin-bottom: 1.2rem !important;
  margin-top: -0.3rem !important;
}
.stSelectbox label, .stRadio label, .stSlider label, h2, h3 { 
    font-weight: 700; color: #213052;
}
.stRadio, .stSelectbox, .stSlider {margin-bottom: 0.15rem;}
.stRadio, .stSelectbox, .stSlider > label {padding-top: 0rem !important; margin-bottom:0rem !important;}
footer {margin-top: 1.0rem;}
hr {border-top: 1px solid #dbe2ee;}
/* Remove excess vertical gap between columns on wide screens */
[data-testid="column"] {padding-top: 6px!important;}
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_resource
def load_panel_data():
    df_CES = pl.read_parquet("CES_2015_2026_01_14.parquet")
    df_CEU = pl.read_parquet("CEU_2015_2026_01_14.parquet")
    return df_CES, df_CEU
df_CES, df_CEU = load_panel_data()

def extract_metric_name(series_title, sector=None, industry=None):
    if series_title is None or (isinstance(series_title, float) and pl.is_nan(series_title)): return ""
    parts = [p.strip() for p in str(series_title).split(',')]
    # Remove seasonality
    if parts and parts[-1].lower() in ["seasonally adjusted", "not seasonally adjusted"]: parts = parts[:-1]
    if len(parts) > 1 and (
        (sector and parts[-1].lower() == sector.lower())
        or (industry and parts[-1].lower() == industry.lower())
    ): parts = parts[:-1]
    return ', '.join(parts)

def pretty_industry(sector, industry):
    if not industry: return ""
    return f"{industry} (All of sector)" if sector == industry else str(industry)

def get_metrics_for_combo(df, sector, industry):
    if not sector or not industry: return set(), {}
    mask = (
        pl.col("Commerce_Sector").str.strip_chars().str.to_lowercase() == sector.lower().strip()
    ) & (
        pl.col("Commerce_Industry").str.strip_chars().str.to_lowercase() == industry.lower().strip()
    )
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
    result = {}
    for s in shared:
        for m in metric_maps:
            if s in m: result[s] = m[s]; break
    return result

st.title("CES/CEU Multi-Series Plotter")

col1, col2 = st.columns(2, gap="large")

# ======= Chart 1: Compact Card + Polars ==============
with col1:
    with st.container():
        st.markdown("### National-Level CES/CEU Trends by Sector & Industry")
        st.caption("Choose sector, industry, and metric to visualize employment trends below.")

        row_display1, row_season1 = st.columns(2)
        with row_display1:
            series_type_1 = st.radio("Display as (Chart 1):", ["Employment Level (Total)", "Monthly Change"], index=0, horizontal=True, key="series_type_1")
        with row_season1:
            seasonal_file_1 = st.radio("Seasonality (Chart 1):", ["Seasonally Adjusted", "Not Seasonally Adjusted"], index=0, horizontal=True, key="season1")
        level_mode_1 = (series_type_1 == "Employment Level (Total)")
        df_1 = df_CES if seasonal_file_1 == "Seasonally Adjusted" else df_CEU

        sectors_1 = df_1["Commerce_Sector"].unique().to_list()
        sectors_1 = sorted(sectors_1, key=lambda s: (s != "Total nonfarm", s))
        sector_1 = st.selectbox("Select sector:", sectors_1, index=0, key="sect1")

        industry_options_1 = df_1.filter(pl.col("Commerce_Sector") == sector_1)["Commerce_Industry"].unique().to_list()
        industry_options_1_sorted = [sector_1] + sorted([ind for ind in industry_options_1 if ind != sector_1])
        industry_options_1_pretty = [pretty_industry(sector_1, ind) for ind in industry_options_1_sorted]
        ind_map_1 = dict(zip(industry_options_1_pretty, industry_options_1_sorted))
        industry_pretty_1 = st.selectbox("Select industry:", industry_options_1_pretty, index=0, key="ind1")
        industry_1 = ind_map_1[industry_pretty_1]

        metrics_1, mapping_1 = get_metrics_for_combo(df_1, sector_1, industry_1)
        main_metric_1 = [m for m in metrics_1 if "all employees, thousands" in m.lower()]
        other_metrics_1 = [m for m in metrics_1 if "all employees, thousands" not in m.lower()]
        metrics_sorted_1 = main_metric_1 + sorted(other_metrics_1)
        
        current_metric_1 = st.session_state.get("ser1", None)
        if current_metric_1 in metrics_sorted_1:
            metric_default_value = current_metric_1
        elif any(m.lower().startswith("all employees, thousands") for m in metrics_sorted_1):
            metric_default_value = next(m for m in metrics_sorted_1 if m.lower().startswith("all employees, thousands"))
        else:
            metric_default_value = metrics_sorted_1[0] if metrics_sorted_1 else None
        
        if metric_default_value is not None and metric_default_value in metrics_sorted_1:
            metric_default_index = metrics_sorted_1.index(metric_default_value)
        else:
            metric_default_index = 0
        
        if metrics_sorted_1:
            metric_choice_1 = st.selectbox(
                "Select metric to plot:",
                metrics_sorted_1,
                index=metric_default_index,
                key="ser1"
            )
            series_title_1 = mapping_1[metric_choice_1]
        else:
            metric_choice_1, series_title_1 = None, None

        dates1 = df_1["Date"].sort().unique().to_list()
        min_date1, max_date1 = dates1[0], dates1[-1]
        min_year1, max_year1 = min_date1.year, max_date1.year

        if level_mode_1:
            default_start_year1 = max(min_year1, max_year1 - 5)
            default_slider1 = (default_start_year1, max_year1)
        else:
            default_start_year1 = max(max_year1 - 1, 2022)
            default_slider1 = (default_start_year1, max_year1)

        year_range1 = st.slider("Select time range:", min_value=min_year1, max_value=max_year1, value=default_slider1, label_visibility="collapsed", key="slider1")
        start_date1 = pl.datetime(year_range1[0], 1, 1)
        end_date1 = pl.datetime(year_range1[1], 12, 31)

        # Filtering is all Polars, only to_pandas for plotting
        if series_title_1:
            plot_df_1 = (
                df_1.filter(
                    (pl.col("Commerce_Sector").str.strip_chars().str.to_lowercase() == sector_1.lower().strip()) &
                    (pl.col("Commerce_Industry").str.strip_chars().str.to_lowercase() == industry_1.lower().strip()) &
                    (pl.col("Series_Title") == series_title_1) &
                    (pl.col("Date") >= start_date1) &
                    (pl.col("Date") <= end_date1)
                )
                .sort("Date")
            )
        else:
            plot_df_1 = pl.DataFrame()
        # Monthly change handling
        if not level_mode_1 and plot_df_1.height > 0:
            plot_df_1 = (
                plot_df_1.with_columns(
                    Value=pl.col("Value").diff()
                )
                .drop_nulls("Value")
            )

        if plot_df_1.height > 0:
            plot_pd_1 = plot_df_1.with_columns([
                pl.lit(pretty_industry(sector_1, industry_1)).alias("Legend"),
            ]).to_pandas()
            value_label_1 = f"{metric_choice_1}{'' if level_mode_1 else ' (MoM Change)'}"
            if level_mode_1:
                fig = px.line(plot_pd_1, x='Date', y='Value', color='Legend',
                             labels={'Value': value_label_1, 'Date': '', 'Legend': ''})
                fig.update_traces(line_width=2)
            else:
                fig = px.bar(plot_pd_1, x='Date', y='Value', color='Legend', barmode="group",
                             labels={'Value': value_label_1, 'Date': '', 'Legend': ''})
            fig.update_layout(
                font=dict(color="black"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.25,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="black"),
                    title=""
                ),
                xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
                yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=520,
                margin=dict(l=20, r=20, t=100, b=40),
                hovermode='x unified',
                
                title=f"<b>{pretty_industry(sector_1, industry_1)}</b> — {seasonal_file_1}<br><span style='font-size:13px;color:gray;'>Sector: {sector_1}</span><br><span style='font-size:15px;'>{metric_choice_1}{'' if level_mode_1 else ' (Monthly Change)'}</span>"
            )
            for trace in fig.data:
                trace.hovertemplate = f'<b>{trace.name}</b><br>%{{x|%b %Y}}: %{{y:,.0f}}<extra></extra>'
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select an available combination and metric to begin.")

# ======= Chart 2: Compact Card + Polars ==============
with col2:
    with st.container():
        st.markdown("### Flexible Compare: Up to 3 Sectors/Industries")
        st.caption("Compare up to three CES/CEU sector/industry metrics (same time range and metric across all).")

        row_display2, row_season2 = st.columns(2)
        with row_display2:
            series_type_2 = st.radio("Display as (Compare Chart):", ["Employment Level (Total)", "Monthly Change"], index=0, horizontal=True, key="series_type_2")
        with row_season2:
            seasonal_file_2 = st.radio("Seasonality (Compare Chart):", ["Seasonally Adjusted", "Not Seasonally Adjusted"], index=0, horizontal=True, key="season2")
        level_mode_2 = (series_type_2 == "Employment Level (Total)")
        df_2 = df_CES if seasonal_file_2 == "Seasonally Adjusted" else df_CEU

        SECTOR_CHOICES = 3
        all_sectors = sorted(df_2["Commerce_Sector"].unique().to_list(), key=lambda s: (s != "Total nonfarm", s))
        sector_columns = st.columns(SECTOR_CHOICES, gap="small")
        industry_columns = st.columns(SECTOR_CHOICES, gap="small")

        defaults = [("Total nonfarm", "Total nonfarm"), ("Total private", "Total private"), ("", "")]
        sector_selections = []
        for i, col in enumerate(sector_columns):
            default_sector = defaults[i][0]
            sector = col.selectbox(f"Sector {i+1}", options=[""] + all_sectors, index=all_sectors.index(default_sector)+1 if default_sector else 0, key=f"cmp_sect{i}")
            sector_selections.append(sector if sector else None)
        industry_selections = []
        ind_pretty_selections = []
        for i, (sect, col) in enumerate(zip(sector_selections, industry_columns)):
            if sect:
                inds = df_2.filter(pl.col("Commerce_Sector") == sect)["Commerce_Industry"].unique().to_list()
                inds_sorted = [sect] + sorted([ind for ind in inds if ind != sect])
                inds_pretty = [pretty_industry(sect, ind) for ind in inds_sorted]
                if defaults[i][1] and sect == defaults[i][0]:
                    default_ind = inds_sorted.index(defaults[i][1]) if defaults[i][1] in inds_sorted else 0
                else:
                    default_ind = 0
                pretty = col.selectbox(f"Industry {i+1}", options=[""] + inds_pretty, index=default_ind+1 if default_ind is not None else 0, key=f"cmp_ind{i}")
                if pretty:
                    ind_map = dict(zip(inds_pretty, inds_sorted))
                    ind_sel = ind_map[pretty]
                    industry_selections.append(ind_sel)
                    ind_pretty_selections.append(pretty)
                else:
                    industry_selections.append(None)
                    ind_pretty_selections.append("")
            else:
                industry_selections.append(None)
                ind_pretty_selections.append("")
        chosen_combos = [
            (sector_selections[i], industry_selections[i], ind_pretty_selections[i])
            for i in range(SECTOR_CHOICES)
            if sector_selections[i] and industry_selections[i]
        ]
        if chosen_combos:
            metric_maps = []
            for (sect, ind, _) in chosen_combos:
                _, m = get_metrics_for_combo(df_2, sect, ind)
                metric_maps.append(m)
            common_metrics_map = intersection_maps(metric_maps)
            main_metrics = [m for m in common_metrics_map.keys() if "all employees, thousands" in m.lower()]
            other_metrics = [m for m in common_metrics_map.keys() if "all employees, thousands" not in m.lower()]
            metrics_sorted = main_metrics + sorted(other_metrics)
        else:
            metrics_sorted = []
            common_metrics_map = {}

        # Only show metric selectbox ONCE, here:
        metric_choice = None
        if metrics_sorted:
            current_metric_2 = st.session_state.get("metric_2", None)
            if current_metric_2 in metrics_sorted:
                metric_default_value_2 = current_metric_2
            elif any(m.lower().startswith("all employees, thousands") for m in metrics_sorted):
                metric_default_value_2 = next(m for m in metrics_sorted if m.lower().startswith("all employees, thousands"))
            else:
                metric_default_value_2 = metrics_sorted[0]
            if metric_default_value_2 is not None and metric_default_value_2 in metrics_sorted:
                metric_default_index_2 = metrics_sorted.index(metric_default_value_2)
            else:
                metric_default_index_2 = 0
            metric_choice = st.selectbox(
                "Select metric to plot (in all series):",
                metrics_sorted,
                index=metric_default_index_2,
                key="metric_2"
            )
        else:
            st.info("Select at least one sector/industry pair with a common metric.")

        dates2 = df_2["Date"].sort().unique().to_list()
        min_date2, max_date2 = dates2[0], dates2[-1]
        min_year2, max_year2 = min_date2.year, max_date2.year

        if level_mode_2:
            default_start_year2 = max(min_year2, max_year2 - 5)
            default_slider2 = (default_start_year2, max_year2)
        else:
            default_start_year2 = max(max_year2 - 1, 2022)
            default_slider2 = (default_start_year2, max_year2)
        year_range2 = st.slider("Select time range (compare chart):", min_value=min_year2, max_value=max_year2, value=default_slider2, label_visibility="collapsed", key="slider2")
        start_date2 = pl.datetime(year_range2[0], 1, 1)
        end_date2 = pl.datetime(year_range2[1], 12, 31)

        def generate_chart2_plot():
            frames = []
            for (sect, ind, ind_pretty) in chosen_combos:
                _, metric_map = get_metrics_for_combo(df_2, sect, ind)
                if metric_choice in metric_map:
                    fulltitle = metric_map[metric_choice]
                    tmp = (
                        df_2.filter(
                            (pl.col("Commerce_Sector").str.strip_chars().str.to_lowercase() == sect.lower().strip()) &
                            (pl.col("Commerce_Industry").str.strip_chars().str.to_lowercase() == ind.lower().strip()) &
                            (pl.col("Series_Title") == fulltitle) &
                            (pl.col("Date") >= start_date2) &
                            (pl.col("Date") <= end_date2)
                        )
                        .sort("Date")
                    )
                    if not level_mode_2 and tmp.height > 0:
                        tmp = (
                            tmp.with_columns(Value=pl.col("Value").diff())
                            .drop_nulls("Value")
                        )
                    if tmp.height > 0:
                        tmp = tmp.with_columns([
                            pl.lit(pretty_industry(sect, ind)).alias("Legend")
                        ])
                        frames.append(tmp)
            plot_df2 = pl.concat(frames) if frames else pl.DataFrame()
            plot_title = f"<b>Compare Series</b> — {seasonal_file_2}<br><span style='font-size:15px;'>{metric_choice}{'' if level_mode_2 else ' (Monthly Change)'}</span>"
            return plot_df2, plot_title

        plot_df, plot_title = generate_chart2_plot()
        if plot_df.height > 0 and metrics_sorted:
            plot_pd = plot_df.to_pandas()
            color_palette = px.colors.qualitative.Set2 + px.colors.qualitative.Plotly + px.colors.qualitative.D3
            unique_legends = plot_pd["Legend"].unique()
            while len(color_palette) < len(unique_legends): color_palette += color_palette
            color_discrete_map = dict(zip(unique_legends, color_palette[:len(unique_legends)]))
            value_label_2 = f"{metric_choice}{'' if level_mode_2 else ' (MoM Change)'}"
            if level_mode_2:
                fig = px.line(plot_pd, x='Date', y='Value', color='Legend', color_discrete_map=color_discrete_map, labels={'Value': value_label_2, 'Legend': '', 'Date': ''})
                fig.update_traces(line_width=2)
            else:
                fig = px.bar(plot_pd, x='Date', y='Value', color='Legend', barmode="group", color_discrete_map=color_discrete_map, labels={'Value': value_label_2, 'Legend': '', 'Date': ''})
          
            fig.update_layout(
                font=dict(color="black"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.25,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="black"),
                    title=""
                ),
                xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
                yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=520,
                margin=dict(l=20, r=20, t=100, b=40),
                hovermode='x unified',
                title=plot_title
            )
            for trace in fig.data:
                trace.hovertemplate = f'<b>{trace.name}</b><br>%{{x|%b %Y}}: %{{y:,.0f}}<extra></extra>'
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for the combination selected.")

st.markdown("<br><span style='color: gray; font-size: 13px;'>Source: U.S. Bureau of Labor Statistics, CES/CEU (Current Employment Statistics)</span>", unsafe_allow_html=True)

import plotly.graph_objs as go


# ---- Cache sector-industry maps and pretty names on first load ----
if "sector_choices" not in st.session_state:
    st.session_state.sector_choices = sorted(
        df_CES["Commerce_Sector"].unique().to_list(), 
        key=lambda s: (s != "Total nonfarm", s)
    )
if "sector_industry_map" not in st.session_state:
    st.session_state.sector_industry_map = {
        s: df_CES.filter(pl.col("Commerce_Sector") == s)["Commerce_Industry"].unique().to_list()
        for s in st.session_state.sector_choices
    }
if "pretty_names_map" not in st.session_state:
    st.session_state.pretty_names_map = {}
    for sector in st.session_state.sector_choices:
        for ind in st.session_state.sector_industry_map[sector]:
            st.session_state.pretty_names_map[(sector, ind)] = pretty_industry(sector, ind)

# ---- Default sector/industries for first load ----
default_sectors_chart3 = [
    "Manufacturing",
    "Construction",
    "Mining and logging",
    "Transportation and warehousing",
    "Utilities"
]
if "chart3_sectors" not in st.session_state or not st.session_state.chart3_sectors:
    st.session_state.chart3_sectors = []
    for s in default_sectors_chart3:
        inds_sorted = [s] + sorted([ind for ind in st.session_state.sector_industry_map[s] if ind != s])
        sector_total_pretty = pretty_industry(s, s)
        st.session_state.chart3_sectors.append({"sector": s, "industries": [sector_total_pretty]})
if "chart3_last_sectors" not in st.session_state or not st.session_state.chart3_last_sectors:
    st.session_state.chart3_last_sectors = [d["sector"] for d in st.session_state.chart3_sectors]

col3, _ = st.columns(2, gap="large")
col3, _ = st.columns(2, gap="large")
with col3:
    # ---- Card & Section styling ----
    st.markdown(
        "<h3>📊 <b>Monthly Year-over-Year Change (Stacked Bar)</b></h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class='card-caption'>
        Pre-populated with: <b>Manufacturing, Construction, Mining and logging, Transportation and warehousing, Utilities</b>.
        <br>
        <span style='color:#5d6e85;'>A dashed black line shows total employment growth/net change (negatives subtract).
        <br>
        <b>➕</b> to add a sector, <b>❌</b> to remove (see below).</span>
        </div>""", unsafe_allow_html=True
    )
    st.divider()

    # --- Deletion and Render Patch ---
    remove_sector_key = "remove_sector_idx"
    if st.session_state.get(remove_sector_key, None) is not None:
        idx_to_remove = st.session_state[remove_sector_key]
        # Only delete if more than 1 sector left and the index is valid
        if 0 <= idx_to_remove < len(st.session_state.chart3_sectors) and len(st.session_state.chart3_sectors) > 1:
            del st.session_state.chart3_sectors[idx_to_remove]
            del st.session_state.chart3_last_sectors[idx_to_remove]
            # Remove associated industry selection key
            rem_key = f'c3_multi_{idx_to_remove}'
            if rem_key in st.session_state:
                del st.session_state[rem_key]
        # Reset the trigger
        st.session_state[remove_sector_key] = None
        st.rerun()

    cols = st.columns(len(st.session_state.chart3_sectors))
    updated_sectors = []
    updated_last_sectors = []
    for idx, c in enumerate(cols):
        with c:
            prev_sector = st.session_state.chart3_sectors[idx]["sector"]
            prev_inds = st.session_state.chart3_sectors[idx]["industries"]
            prev_last_sector = (
                st.session_state.chart3_last_sectors[idx]
                if idx < len(st.session_state.chart3_last_sectors)
                else prev_sector
            )
            sector = st.selectbox(
                f"Sector {idx+1}",
                [""] + st.session_state.sector_choices,
                index=([""] + st.session_state.sector_choices).index(prev_sector) if prev_sector in [""] + st.session_state.sector_choices else 0,
                key=f"c3_sector_{idx}"
            )

            if sector:
                inds_sorted = [sector] + sorted([ind for ind in st.session_state.sector_industry_map[sector] if ind != sector])
                inds_pretty = [pretty_industry(sector, ind) for ind in inds_sorted]
                sector_total_pretty = pretty_industry(sector, sector)
                default_inds = st.session_state.get(f"c3_multi_{idx}", None)
                if sector != prev_last_sector or not prev_inds:
                    industries_selected = default_inds or [sector_total_pretty]
                else:
                    industries_selected = default_inds if default_inds is not None else prev_inds
                industries_selected = [ind for ind in industries_selected if ind in inds_pretty]
                industries_selected = st.multiselect(
                    f"Industries for {sector}",
                    options=inds_pretty,
                    default=industries_selected,
                    key=f"c3_multi_{idx}"
                )
            else:
                inds_pretty = []
                industries_selected = []

            # REMOVE BUTTON: Sets a removal trigger, which is processed next run
            if st.button("❌", key=f"c3remove_{idx}"):
                st.session_state[remove_sector_key] = idx
                st.rerun()

            updated_sectors.append({"sector": sector, "industries": industries_selected})
            updated_last_sectors.append(sector)

    # ---- Add sector column (new starts blank) ----
    bcols = st.columns([1, 6])
    with bcols[0]:
        if st.button("➕", help="Add another sector column", key="c3add") and len(st.session_state.chart3_sectors) < 8:
            st.session_state.chart3_sectors.append({"sector": "", "industries": []})
            st.session_state.chart3_last_sectors.append("")
            st.rerun()

    # Update session state for the next run
    st.session_state.chart3_sectors = updated_sectors
    st.session_state.chart3_last_sectors = updated_last_sectors

    # Prepare sector/industry pairs for plotting
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

    # ---- Metric and seasonal selection ----
    metric_maps = []
    for sect, ind, _ in selected:
        _, m = get_metrics_for_combo(df_CES, sect, ind)
        metric_maps.append(m)
    common_metrics_map = intersection_maps(metric_maps) if metric_maps else {}
    main_metrics = [m for m in common_metrics_map.keys() if "all employees, thousands" in m.lower()]
    other_metrics = [m for m in common_metrics_map.keys() if "all employees, thousands" not in m.lower()]
    metrics_sorted = main_metrics + sorted(other_metrics)

    optcols = st.columns(2)
    with optcols[0]:
        if metrics_sorted:
            preferred = "all employees, thousands"
            default_idx = next(
                (metrics_sorted.index(m) for m in metrics_sorted if m.lower() == preferred), 
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
        source_file = st.radio(
            "Data set:",
            ["Seasonally Adjusted", "Not Seasonally Adjusted"],
            index=0,
            horizontal=True,
            key="ct3_season"
        )
    df_3 = df_CES if source_file == "Seasonally Adjusted" else df_CEU

    # ---- Slider: default start at 2022 or min_year ----
    if len(df_3) > 0:
        min_date3 = df_3["Date"].min()
        max_date3 = df_3["Date"].max()
        min_year3 = min_date3.year
        max_year3 = max_date3.year
        slider_start = max(min_year3, 2022)
        year_range3 = st.slider(
            "Year range:",
            min_value=min_year3,
            max_value=max_year3,
            value=(slider_start, max_year3),
            step=1,
            key="ct3_years"
        )
        date_start = pl.datetime(year_range3[0], 1, 1)
        date_end = pl.datetime(year_range3[1], 12, 31)
    else:
        date_start = date_end = datetime.now()

    # --- Efficient bulk filter for all selected combos ---
    chart_filters = []
    for sect, ind, legend_label in selected:
        _, metric_map = get_metrics_for_combo(df_3, sect, ind)
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
        mask = (
            (pl.col("Date") >= date_start) &
            (pl.col("Date") <= date_end)
        )
        df_filtered = df_3.filter(mask)
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

    # --- Data and Plot ---
    with st.container():
        if chart3_rows:
            chart3_df = pl.concat(chart3_rows)
            chart3_pd = chart3_df.to_pandas()
            chart3_pd['Month'] = chart3_pd['Date'].dt.to_period('M').astype(str)
            st.markdown(f"""
            <span style='color:#1b293e; font-size:15px;'>Metric: <b>{metric_choice}</b> | Time: <b>{year_range3[0]}–{year_range3[1]}</b> | <b>{source_file}</b></span>
            """, unsafe_allow_html=True)

            fig = px.bar(
                chart3_pd,
                x="Month",
                y="YoY_Change",
                color="Legend",
                labels={"YoY_Change": metric_choice + " (YoY Δ)", "Legend": ""},
                barmode="relative"
            )
            # Add dashed black line: total employment growth per month
            if not chart3_pd.empty:
                total_line_df = (
                    chart3_pd.groupby("Month", as_index=False)
                    ["YoY_Change"].sum()
                    .rename(columns={"YoY_Change": "Total Employment Growth"})
                )
                fig.add_trace(
                    go.Scatter(
                        x=total_line_df["Month"],
                        y=total_line_df["Total Employment Growth"],
                        name="Total Employment Growth",
                        mode="lines+markers",
                        marker=dict(color="black", size=7),
                        line=dict(color="black", width=3, dash="dash"),
                        hovertemplate="<b>Total Employment Growth</b><br>Month: %{x}<br>Growth: %{y:,.0f}<extra></extra>"
                    )
                )
            fig.update_layout(
                font=dict(color="#23263c"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.32,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="#243350"),
                    title=""
                ),
                xaxis=dict(title=""),
                yaxis=dict(title_font=dict(color="#23263c"), tickfont=dict(color="#23263c")),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=520,
                margin=dict(l=18, r=18, t=48, b=46),
                title=None
            )
            st.plotly_chart(fig, use_container_width=True)
        elif selected and not metric_choice:
            st.warning("No common metric found for these selections.")
        else:
            st.info("Add sector/industry columns and select at least one for comparison.")

    st.divider()
