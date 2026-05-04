"""
Supply Chain Stress-Tester
==========================
A production-grade Streamlit application for simulating tariff shocks,
port closures, and geopolitical disruptions across a synthetic supplier network.

Author : Supply Chain Analytics Team
Version: 1.0.0
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================================
# PAGE CONFIG  (must be first Streamlit call)
# ============================================================================
st.set_page_config(
    page_title="Supply Chain Stress-Tester",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# GLOBAL STYLE  – dark, industrial, amber-accent palette
# ============================================================================
STYLE = """
<style>
/* ── imports ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── root tokens ── */
:root {
    --bg:        #0d0f14;
    --surface:   #141720;
    --border:    #232733;
    --amber:     #f0a500;
    --amber-dim: #a87200;
    --red:       #e05252;
    --green:     #52c07a;
    --text:      #dce3f0;
    --muted:     #6b7590;
    --font-mono: 'Space Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
}

/* ── base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── headings ── */
h1,h2,h3,h4 { font-family: var(--font-mono) !important; color: var(--text) !important; }
h1 { letter-spacing: -1px; }

/* ── metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"]  { color: var(--muted) !important; font-size:0.75rem; text-transform:uppercase; letter-spacing:.06em; }
[data-testid="stMetricValue"]  { color: var(--amber) !important; font-family: var(--font-mono) !important; }
[data-testid="stMetricDelta"]  { font-family: var(--font-mono) !important; }

/* ── tabs ── */
[data-testid="stTabs"] button {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: .05em;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.5rem 1.1rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom: 2px solid var(--amber) !important;
}

/* ── sliders, selects, inputs ── */
[data-testid="stSlider"] .st-bq { background: var(--amber) !important; }
.stSelectbox > div > div,
.stMultiSelect > div > div { background: var(--surface) !important; border-color: var(--border) !important; }

/* ── info / warning boxes ── */
.exec-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #141720 100%);
    border: 1px solid var(--amber-dim);
    border-left: 4px solid var(--amber);
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.exec-card h4 { margin-top:0; color: var(--amber); font-size:0.8rem; text-transform:uppercase; letter-spacing:.1em; }
.exec-card p  { margin-bottom:0; line-height:1.7; font-size:0.95rem; }

.risk-pill {
    display:inline-block;
    padding:2px 10px;
    border-radius:999px;
    font-family: var(--font-mono);
    font-size:0.72rem;
    font-weight:700;
}
.risk-high   { background:#3d1414; color:var(--red); }
.risk-medium { background:#3d2e0a; color:var(--amber); }
.risk-low    { background:#0e2e1c; color:var(--green); }

/* ── divider ── */
hr { border-color: var(--border) !important; }

/* ── dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius:8px; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# ============================================================================
# CONSTANTS & LOOK-UPS
# ============================================================================
REGIONS = {
    "East Asia":        ["China", "Japan", "South Korea", "Taiwan", "Vietnam", "Thailand", "Indonesia", "Malaysia"],
    "South Asia":       ["India", "Bangladesh", "Pakistan", "Sri Lanka"],
    "Europe":           ["Germany", "Poland", "Italy", "France", "Czech Republic", "Netherlands"],
    "North America":    ["Mexico", "United States", "Canada"],
    "Middle East":      ["Turkey", "UAE", "Saudi Arabia", "Israel"],
    "Latin America":    ["Brazil", "Colombia", "Chile", "Argentina"],
}

COUNTRY_COORDS = {
    "China":          (35.86, 104.19), "Japan":          (36.20, 138.25),
    "South Korea":    (35.91, 127.77), "Taiwan":         (23.70, 121.00),
    "Vietnam":        (14.06,  108.28),"Thailand":       (15.87, 100.99),
    "Indonesia":      (-0.79, 113.92), "Malaysia":       (4.21,  109.45),
    "India":          (20.59,  78.96), "Bangladesh":     (23.68,  90.36),
    "Pakistan":       (30.38,  69.35), "Sri Lanka":      (7.87,   80.77),
    "Germany":        (51.17,  10.45), "Poland":         (51.92,  19.14),
    "Italy":          (41.87,  12.56), "France":         (46.23,   2.21),
    "Czech Republic": (49.82,  15.47), "Netherlands":    (52.13,   5.29),
    "Mexico":         (23.63, -102.55),"United States":  (37.09, -95.71),
    "Canada":         (56.13, -106.35),"Turkey":         (38.96,  35.24),
    "UAE":            (23.42,  53.85), "Saudi Arabia":   (23.89,  45.08),
    "Israel":         (31.05,  34.85), "Brazil":         (-14.24, -51.93),
    "Colombia":       (4.57,  -74.30), "Chile":          (-35.68, -71.54),
    "Argentina":      (-38.42, -63.62),
}

PORTS = {
    "China":          ["Shanghai", "Shenzhen", "Ningbo"],
    "Japan":          ["Tokyo", "Osaka"],
    "South Korea":    ["Busan"],
    "Taiwan":         ["Kaohsiung"],
    "Vietnam":        ["Ho Chi Minh City", "Hai Phong"],
    "Thailand":       ["Bangkok"],
    "Indonesia":      ["Jakarta"],
    "Malaysia":       ["Port Klang"],
    "India":          ["Mumbai", "Chennai", "Nhava Sheva"],
    "Bangladesh":     ["Chittagong"],
    "Pakistan":       ["Karachi"],
    "Sri Lanka":      ["Colombo"],
    "Germany":        ["Hamburg", "Bremen"],
    "Poland":         ["Gdansk"],
    "Italy":          ["Genoa", "La Spezia"],
    "France":         ["Marseille", "Le Havre"],
    "Czech Republic": ["Prague (Rail)"],
    "Netherlands":    ["Rotterdam"],
    "Mexico":         ["Manzanillo", "Veracruz"],
    "United States":  ["Los Angeles", "New York"],
    "Canada":         ["Vancouver"],
    "Turkey":         ["Istanbul", "Mersin"],
    "UAE":            ["Dubai (Jebel Ali)"],
    "Saudi Arabia":   ["Jeddah"],
    "Israel":         ["Haifa"],
    "Brazil":         ["Santos"],
    "Colombia":       ["Cartagena"],
    "Chile":          ["Valparaiso"],
    "Argentina":      ["Buenos Aires"],
}

CATEGORIES = ["Electronics", "Textiles", "Chemicals", "Automotive", "Machinery",
               "Pharmaceuticals", "Raw Materials", "Consumer Goods"]

# ── Preset Scenarios ─────────────────────────────────────────────────────────
PRESETS = {
    "🇺🇸 US-China Trade War": {
        "description": "Blanket 60% tariff on all Chinese goods; factories shift to Vietnam, India & Mexico.",
        "tariff_overrides": {"East Asia": 60.0, "South Asia": 12.0, "North America": 5.0},
        "closed_ports": ["Shanghai", "Shenzhen"],
        "delay_mult": 1.35,
    },
    "🚢 Red Sea Crisis": {
        "description": "Houthi attacks reroute ships around Cape of Good Hope; +14-day transit, +30% freight.",
        "tariff_overrides": {"Middle East": 8.0, "Europe": 4.0},
        "closed_ports": ["Dubai (Jebel Ali)", "Jeddah"],
        "delay_mult": 1.65,
    },
    "🦠 Pandemic Shock (2020-style)": {
        "description": "Global factory shutdowns, border closures, lead-time doubling, risk scores spike.",
        "tariff_overrides": {},
        "closed_ports": ["Shanghai", "Rotterdam", "Los Angeles", "Mumbai"],
        "delay_mult": 2.10,
    },
    "🇪🇺 EU Carbon Border Adjustment": {
        "description": "CBAM adds 15% effective cost levy on carbon-intensive imports to EU markets.",
        "tariff_overrides": {"Europe": 15.0, "East Asia": 8.0, "South Asia": 6.0},
        "closed_ports": [],
        "delay_mult": 1.10,
    },
    "⚡ Taiwan Strait Escalation": {
        "description": "Taiwan Strait closure reroutes electronics; chipmaker diversification mandatory.",
        "tariff_overrides": {"East Asia": 45.0},
        "closed_ports": ["Kaohsiung", "Shanghai", "Ningbo"],
        "delay_mult": 1.80,
    },
}

# ============================================================================
# SYNTHETIC DATA GENERATOR
# ============================================================================
@st.cache_data(show_spinner=False)
def generate_suppliers(seed: int = 42, n: int = 800) -> pd.DataFrame:
    """Generate a reproducible synthetic supplier dataset."""
    rng = np.random.default_rng(seed)
    records = []
    id_counter = 1

    country_region = {c: r for r, cs in REGIONS.items() for c in cs}
    all_countries  = list(country_region.keys())

    # Weighted towards East Asia & South Asia (realistic)
    weights = []
    for c in all_countries:
        r = country_region[c]
        if r == "East Asia":    weights.append(6)
        elif r == "South Asia": weights.append(4)
        elif r == "Europe":     weights.append(3)
        else:                    weights.append(2)
    wt = np.array(weights, dtype=float)
    wt /= wt.sum()

    for _ in range(n):
        country  = rng.choice(all_countries, p=wt)
        region   = country_region[country]
        port     = rng.choice(PORTS[country])
        category = rng.choice(CATEGORIES)
        lat, lon = COUNTRY_COORDS[country]
        lat += rng.uniform(-2, 2)
        lon += rng.uniform(-2, 2)

        volume    = int(rng.lognormal(mean=9.5, sigma=1.2))          # units/year
        base_cost = round(float(rng.lognormal(mean=5.2, sigma=0.8)), 2)  # USD/unit

        # Tariff reflects realistic 2024 baseline
        tariff_base = {
            "East Asia": 7.5, "South Asia": 5.5, "Europe": 2.5,
            "North America": 0.5, "Middle East": 4.0, "Latin America": 3.5,
        }[region]
        tariff = round(float(rng.normal(tariff_base, 1.5)), 2)
        tariff = max(0.0, tariff)

        lead_time  = int(rng.integers(7, 90))                         # days
        risk_score = round(float(rng.beta(2, 5) * 10), 2)            # 0–10

        records.append({
            "SupplierID":  f"SUP-{id_counter:04d}",
            "Country":     country,
            "Region":      region,
            "Port":        port,
            "Category":    category,
            "Volume":      volume,
            "BaseCost":    base_cost,
            "TariffPct":   tariff,
            "LeadTime":    lead_time,
            "RiskScore":   risk_score,
            "Lat":         round(lat, 4),
            "Lon":         round(lon, 4),
        })
        id_counter += 1

    df = pd.DataFrame(records)
    df["TotalBaseCost"] = (df["BaseCost"] * (1 + df["TariffPct"] / 100) * df["Volume"]).round(2)
    return df


# ============================================================================
# SIMULATION ENGINE
# ============================================================================
def simulate(
    df: pd.DataFrame,
    tariff_delta: dict,          # {region: new_tariff_pct}
    closed_ports: list,
    delay_mult: float,
    mc_runs: int = 1000,
) -> pd.DataFrame:
    """
    Apply stress parameters and run Monte Carlo cost distributions.
    Returns enriched dataframe with stressed columns.
    """
    sim = df.copy()

    # ── Apply tariff overrides by region ─────────────────────────────────
    for region, new_pct in tariff_delta.items():
        mask = sim["Region"] == region
        sim.loc[mask, "TariffPct"] = new_pct

    # ── Port closure → rerouting cost (+25%) & extra delay (+10 days) ────
    port_affected = sim["Port"].isin(closed_ports)
    sim["PortClosed"]    = port_affected
    sim["RerouteCost"]   = np.where(port_affected, sim["BaseCost"] * 0.25, 0.0)
    sim["ExtraLeadDays"] = np.where(port_affected, 10, 0)

    # ── Delay multiplier on lead time ────────────────────────────────────
    sim["StressedLeadTime"] = (sim["LeadTime"] * delay_mult + sim["ExtraLeadDays"]).round(1)

    # ── Stressed total cost ──────────────────────────────────────────────
    sim["TotalStressedCost"] = (
        (sim["BaseCost"] + sim["RerouteCost"])
        * (1 + sim["TariffPct"] / 100)
        * sim["Volume"]
    ).round(2)

    # ── Cost delta & % change ─────────────────────────────────────────────
    sim["CostDelta"]  = (sim["TotalStressedCost"] - sim["TotalBaseCost"]).round(2)
    sim["CostDeltaPct"] = ((sim["CostDelta"] / sim["TotalBaseCost"]) * 100).round(2)

    # ── Monte Carlo on stressed cost (±15% noise) ─────────────────────────
    rng = np.random.default_rng(99)
    noise = rng.normal(1.0, 0.15, size=(mc_runs, len(sim)))   # shape (runs, suppliers)
    mc_total = (sim["TotalStressedCost"].values * noise).sum(axis=1) / 1e6  # $M
    sim.attrs["mc_distribution"] = mc_total   # store on df for retrieval

    # ── Updated risk score: port closure & lead-time spike raise risk ─────
    sim["StressedRisk"] = (
        sim["RiskScore"]
        + port_affected.astype(float) * 2.0
        + np.clip((sim["StressedLeadTime"] - sim["LeadTime"]) / 30, 0, 3)
    ).clip(0, 10).round(2)

    return sim


# ============================================================================
# CHART HELPERS
# ============================================================================
PLOTLY_TEMPLATE = "plotly_dark"
BG   = "#0d0f14"
SURF = "#141720"
AMBER = "#f0a500"
RED   = "#e05252"
GREEN = "#52c07a"

def fig_defaults(fig):
    """Apply consistent dark-mode defaults to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=SURF,
        font=dict(family="DM Sans, sans-serif", color="#dce3f0"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def world_map(df: pd.DataFrame, color_col: str, title: str):
    hover_cols = ["SupplierID", "Country", "Port", "Category", "Volume"]
    fig = px.scatter_geo(
        df, lat="Lat", lon="Lon",
        color=color_col,
        size="Volume",
        size_max=22,
        color_continuous_scale=["#0e2e1c", GREEN, AMBER, RED],
        hover_name="SupplierID",
        hover_data={c: True for c in hover_cols if c in df.columns},
        title=title,
        projection="natural earth",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_geos(
        bgcolor=BG,
        landcolor="#1c2030",
        oceancolor="#0d1220",
        showocean=True,
        coastlinecolor="#2a3050",
        showcoastlines=True,
        countrycolor="#2a3050",
        showframe=False,
    )
    fig.update_layout(
        paper_bgcolor=BG,
        font=dict(family="DM Sans", color="#dce3f0"),
        coloraxis_colorbar=dict(title=color_col, tickfont=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0),
        height=480,
    )
    return fig


def cost_breakdown_bar(df: pd.DataFrame):
    grp = df.groupby("Region").agg(
        BaseCost=("TotalBaseCost", "sum"),
        StressedCost=("TotalStressedCost", "sum"),
    ).reset_index()
    grp["BaseCost"]     /= 1e6
    grp["StressedCost"] /= 1e6

    fig = go.Figure()
    fig.add_bar(name="Baseline", x=grp["Region"], y=grp["BaseCost"],
                marker_color="#3a4870", text=grp["BaseCost"].round(1),
                texttemplate="%{text}M", textposition="outside")
    fig.add_bar(name="Stressed", x=grp["Region"], y=grp["StressedCost"],
                marker_color=AMBER, text=grp["StressedCost"].round(1),
                texttemplate="%{text}M", textposition="outside")
    fig.update_layout(
        barmode="group", title="Cost Breakdown: Baseline vs Stressed (USD M)",
        template=PLOTLY_TEMPLATE, height=380,
        legend=dict(orientation="h", y=1.05),
        yaxis_title="USD Millions",
    )
    return fig_defaults(fig)


def risk_heatmap(df: pd.DataFrame):
    pivot = df.groupby(["Region", "Category"])["StressedRisk"].mean().unstack(fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, "#0e2e1c"], [0.5, AMBER], [1, RED]],
        text=pivot.values.round(1),
        texttemplate="%{text}",
        colorbar=dict(title="Risk (0–10)", tickfont=dict(size=10)),
        zmin=0, zmax=10,
    ))
    fig.update_layout(
        title="Risk Heatmap: Region × Category (stressed)",
        template=PLOTLY_TEMPLATE, height=360,
        xaxis_title="Category", yaxis_title="Region",
    )
    return fig_defaults(fig)


def mc_histogram(mc_dist: np.ndarray, baseline_total_m: float):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=mc_dist, nbinsx=60,
        marker_color=AMBER, opacity=0.8, name="Monte Carlo runs",
    ))
    fig.add_vline(x=mc_dist.mean(), line_color=RED, line_width=2,
                  annotation_text=f"Mean ${mc_dist.mean():.1f}M",
                  annotation_font_color=RED)
    fig.add_vline(x=baseline_total_m, line_color=GREEN, line_width=2, line_dash="dash",
                  annotation_text=f"Baseline ${baseline_total_m:.1f}M",
                  annotation_font_color=GREEN)
    fig.add_vline(x=np.percentile(mc_dist, 95), line_color="#cc4444", line_width=1.5, line_dash="dot",
                  annotation_text="P95", annotation_font_color="#cc4444")
    fig.update_layout(
        title=f"Monte Carlo Distribution ({len(mc_dist):,} runs) — Total Stressed Cost",
        template=PLOTLY_TEMPLATE, height=360,
        xaxis_title="Total Cost (USD M)", yaxis_title="Frequency",
        showlegend=False,
    )
    return fig_defaults(fig)


def tornado_chart(df: pd.DataFrame):
    """Sensitivity: which regions drive the most cost change?"""
    grp = df.groupby("Region")["CostDelta"].sum().sort_values() / 1e6
    colors = [RED if v > 0 else GREEN for v in grp.values]
    fig = go.Figure(go.Bar(
        x=grp.values, y=grp.index,
        orientation="h",
        marker_color=colors,
        text=[f"${v:+.1f}M" for v in grp.values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Tornado Chart — Cost Delta by Region (USD M)",
        template=PLOTLY_TEMPLATE, height=360,
        xaxis_title="Cost Change (USD M)", yaxis_title="",
        xaxis=dict(zeroline=True, zerolinecolor="#555"),
    )
    return fig_defaults(fig)


# ============================================================================
# SIDEBAR
# ============================================================================
def render_sidebar(df: pd.DataFrame):
    st.sidebar.markdown(
        "<h2 style='font-family:Space Mono;font-size:1.1rem;color:#f0a500;margin-bottom:0'>⚙️ What-If Controls</h2>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    # ── Preset Scenarios ─────────────────────────────────────────────────
    st.sidebar.markdown("**Preset Scenarios**")
    preset_names = ["— Custom —"] + list(PRESETS.keys())
    chosen_preset = st.sidebar.selectbox("Load scenario", preset_names, index=0)

    preset_data = None
    if chosen_preset != "— Custom —":
        preset_data = PRESETS[chosen_preset]
        st.sidebar.info(preset_data["description"])

    st.sidebar.markdown("---")

    # ── Tariff Overrides by Region ────────────────────────────────────────
    st.sidebar.markdown("**Tariff Shock by Region (%)**")
    tariff_delta = {}
    for region in REGIONS.keys():
        default_val = df[df["Region"] == region]["TariffPct"].mean()
        if preset_data and region in preset_data.get("tariff_overrides", {}):
            default_val = preset_data["tariff_overrides"][region]
        val = st.sidebar.slider(
            region, 0.0, 100.0,
            value=float(round(default_val, 1)),
            step=0.5,
            key=f"tariff_{region}",
        )
        tariff_delta[region] = val

    st.sidebar.markdown("---")

    # ── Port Closures ─────────────────────────────────────────────────────
    st.sidebar.markdown("**Port Closures**")
    all_ports = sorted(df["Port"].unique().tolist())
    default_ports = preset_data["closed_ports"] if preset_data else []
    closed_ports = st.sidebar.multiselect(
        "Select closed ports", all_ports, default=default_ports
    )

    st.sidebar.markdown("---")

    # ── Delay Multiplier ─────────────────────────────────────────────────
    st.sidebar.markdown("**Lead-Time Delay Multiplier**")
    default_dm = preset_data["delay_mult"] if preset_data else 1.0
    delay_mult = st.sidebar.slider("×  current lead time", 1.0, 3.0, value=float(default_dm), step=0.05)

    st.sidebar.markdown("---")

    # ── Monte Carlo ───────────────────────────────────────────────────────
    st.sidebar.markdown("**Monte Carlo Simulation**")
    mc_runs = st.sidebar.selectbox("Number of runs", [500, 1000, 2000, 5000], index=1)

    st.sidebar.markdown("---")
    st.sidebar.caption("💡 Adjust controls above and results update instantly.")

    return tariff_delta, closed_ports, delay_mult, mc_runs


# ============================================================================
# TABS
# ============================================================================

# ── TAB: Overview ─────────────────────────────────────────────────────────
def tab_overview(df: pd.DataFrame):
    st.markdown("## 📊 Supplier Network Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Suppliers",  f"{len(df):,}")
    c2.metric("Countries",        f"{df['Country'].nunique()}")
    c3.metric("Baseline Cost",    f"${df['TotalBaseCost'].sum()/1e6:.1f}M")
    c4.metric("Avg Risk Score",   f"{df['RiskScore'].mean():.2f} / 10")

    st.markdown("---")
    col_map, col_dist = st.columns([3, 2])
    with col_map:
        st.plotly_chart(world_map(df, "RiskScore", "Supplier Network — Baseline Risk Score"),
                        use_container_width=True)
    with col_dist:
        fig = px.box(
            df, x="Region", y="TariffPct", color="Region",
            color_discrete_sequence=px.colors.qualitative.Dark24,
            template=PLOTLY_TEMPLATE, title="Tariff % Distribution by Region",
        )
        fig.update_layout(showlegend=False, height=480)
        st.plotly_chart(fig_defaults(fig), use_container_width=True)

    st.markdown("---")
    col_cat, col_lt = st.columns(2)
    with col_cat:
        grp = df.groupby("Category")["TotalBaseCost"].sum().reset_index()
        grp["TotalBaseCost"] /= 1e6
        fig = px.bar(grp.sort_values("TotalBaseCost"),
                     x="TotalBaseCost", y="Category", orientation="h",
                     color="TotalBaseCost", color_continuous_scale=["#1a3a5c", AMBER],
                     template=PLOTLY_TEMPLATE, title="Baseline Cost by Category (USD M)")
        fig.update_layout(showlegend=False, height=340, coloraxis_showscale=False)
        st.plotly_chart(fig_defaults(fig), use_container_width=True)
    with col_lt:
        fig = px.histogram(df, x="LeadTime", color="Region",
                           color_discrete_sequence=px.colors.qualitative.Dark24,
                           template=PLOTLY_TEMPLATE, title="Lead-Time Distribution (days)",
                           barmode="overlay", opacity=0.7)
        fig.update_layout(height=340, xaxis_title="Lead Time (days)")
        st.plotly_chart(fig_defaults(fig), use_container_width=True)


# ── TAB: Simulation ───────────────────────────────────────────────────────
def tab_simulation(df_base: pd.DataFrame, sim: pd.DataFrame, mc_runs: int):
    mc_dist         = sim.attrs["mc_distribution"]
    baseline_total  = df_base["TotalBaseCost"].sum() / 1e6
    stressed_total  = sim["TotalStressedCost"].sum()  / 1e6
    delta_pct       = (stressed_total - baseline_total) / baseline_total * 100
    p95             = np.percentile(mc_dist, 95)
    port_affected   = sim["PortClosed"].sum()

    # ── KPIs ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Baseline Cost",    f"${baseline_total:.1f}M")
    c2.metric("Stressed Cost",    f"${stressed_total:.1f}M",  delta=f"{stressed_total-baseline_total:+.1f}M")
    c3.metric("Cost Δ",           f"{delta_pct:+.1f}%")
    c4.metric("P95 (Monte Carlo)",f"${p95:.1f}M")
    c5.metric("Ports Affected",   f"{port_affected}")

    st.markdown("---")

    # ── World map (stressed risk) ─────────────────────────────────────────
    st.plotly_chart(world_map(sim, "StressedRisk", "Supplier Network — Stressed Risk Score"),
                    use_container_width=True)

    st.markdown("---")
    col_bar, col_tornado = st.columns(2)
    with col_bar:
        st.plotly_chart(cost_breakdown_bar(sim), use_container_width=True)
    with col_tornado:
        st.plotly_chart(tornado_chart(sim), use_container_width=True)

    st.markdown("---")
    col_heat, col_mc = st.columns(2)
    with col_heat:
        st.plotly_chart(risk_heatmap(sim), use_container_width=True)
    with col_mc:
        st.plotly_chart(mc_histogram(mc_dist, baseline_total), use_container_width=True)


# ── TAB: Recommendations ──────────────────────────────────────────────────
def tab_recommendations(df_base: pd.DataFrame, sim: pd.DataFrame):
    baseline_total = df_base["TotalBaseCost"].sum() / 1e6
    stressed_total = sim["TotalStressedCost"].sum()  / 1e6
    delta_pct      = (stressed_total - baseline_total) / baseline_total * 100
    mc_dist        = sim.attrs["mc_distribution"]
    p95            = np.percentile(mc_dist, 95)
    avg_risk       = sim["StressedRisk"].mean()
    worst_region   = sim.groupby("Region")["CostDelta"].sum().idxmax()
    worst_delta    = sim.groupby("Region")["CostDelta"].sum().max() / 1e6
    safest_region  = sim.groupby("Region")["StressedRisk"].mean().idxmin()

    # ── Executive Summary Card ─────────────────────────────────────────────
    severity = "🔴 HIGH" if delta_pct > 20 else ("🟡 MEDIUM" if delta_pct > 8 else "🟢 LOW")
    st.markdown(f"""
    <div class="exec-card">
        <h4>📋 Executive Summary</h4>
        <p>
        Under the applied stress scenario, total supply chain costs rise from
        <strong>${baseline_total:.1f}M</strong> to <strong>${stressed_total:.1f}M</strong>
        — a <strong>{delta_pct:+.1f}%</strong> increase. Monte Carlo simulation (1,000 runs)
        places the 95th-percentile outcome at <strong>${p95:.1f}M</strong>, representing a
        <strong>${p95 - baseline_total:.1f}M tail-risk exposure</strong>.
        Average network risk score climbs to <strong>{avg_risk:.2f}/10</strong>.
        <br><br>
        Scenario severity: <strong>{severity}</strong>. The most impacted region is
        <strong>{worst_region}</strong> (+${worst_delta:.1f}M incremental cost).
        <strong>{safest_region}</strong> presents the lowest stressed-risk profile and
        should be prioritised for near-term sourcing diversification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Strategic Recommendations ─────────────────────────────────────────
    st.markdown("### 🧭 Strategic Actions")

    recs = [
        ("🔀 Diversify Away from High-Tariff Regions",
         f"Shift {min(20 + int(delta_pct), 40)}% of volume from **{worst_region}** to lower-tariff "
         f"alternatives. {safest_region} suppliers can absorb incremental demand with risk scores "
         f"averaging {sim[sim['Region']==safest_region]['StressedRisk'].mean():.1f}/10."),

        ("🛳️ Dual-Port Routing Strategy",
         f"{sim['PortClosed'].sum()} suppliers are exposed to closed ports. Implement dual-port "
         f"qualification requiring each critical supplier to maintain routing via at least two "
         f"major ports to eliminate single points of failure."),

        ("📦 Strategic Inventory Buffers",
         f"With average stressed lead times of **{sim['StressedLeadTime'].mean():.0f} days** "
         f"(vs baseline {sim['LeadTime'].mean():.0f} days), increase safety stock to 8–12 weeks "
         f"for high-volume, high-risk category suppliers."),

        ("📜 Contract Renegotiation with Force Majeure",
         "Update supplier contracts to include tariff pass-through caps at ±15%, force majeure "
         "clauses covering port closures, and automatic rerouting SLAs with defined cost-sharing."),

        ("🤝 Preferred Supplier Development",
         f"Accelerate qualification of suppliers in **{safest_region}** (avg risk {sim[sim['Region']==safest_region]['StressedRisk'].mean():.1f}/10). "
         "Offer 12-month volume guarantees in exchange for priority capacity allocation during disruptions."),

        ("💰 Financial Hedging",
         f"With P95 tail risk at ${p95:.1f}M vs baseline ${baseline_total:.1f}M, consider "
         f"supply chain disruption insurance or parametric hedging instruments covering "
         f"${(p95 - baseline_total)*0.6:.1f}–${(p95 - baseline_total):.1f}M of exposure."),
    ]

    for i, (title, body) in enumerate(recs):
        with st.expander(f"  {i+1}. {title}", expanded=(i < 2)):
            st.markdown(body)

    st.markdown("---")
    st.markdown("### 📋 Top 20 Highest-Risk Suppliers (Stressed)")
    top_risk = (
        sim.sort_values("StressedRisk", ascending=False)
        .head(20)[["SupplierID", "Country", "Port", "Category", "Volume",
                   "BaseCost", "TariffPct", "StressedLeadTime", "StressedRisk",
                   "TotalBaseCost", "TotalStressedCost", "CostDeltaPct"]]
        .reset_index(drop=True)
    )
    top_risk.columns = ["ID", "Country", "Port", "Category", "Vol",
                        "Base$/u", "Tariff%", "Lead(d)", "Risk",
                        "Base$Total", "Stress$Total", "Δ%"]
    st.dataframe(
        top_risk.style
            .background_gradient(subset=["Risk"], cmap="RdYlGn_r")
            .background_gradient(subset=["Δ%"], cmap="RdYlGn_r")
            .format({"Base$/u": "${:.2f}", "Tariff%": "{:.1f}%",
                     "Lead(d)": "{:.0f}", "Risk": "{:.2f}",
                     "Base$Total": "${:,.0f}", "Stress$Total": "${:,.0f}",
                     "Δ%": "{:+.1f}%"}),
        use_container_width=True,
        height=420,
    )

    # ── Sourcing Shift Table ───────────────────────────────────────────────
    st.markdown("### 🌍 Sourcing Shift Recommendations")
    shift_data = []
    for region in sim["Region"].unique():
        r_sim = sim[sim["Region"] == region]
        base_vol  = df_base[df_base["Region"] == region]["Volume"].sum()
        stress_ch = r_sim["CostDeltaPct"].mean()
        risk_avg  = r_sim["StressedRisk"].mean()
        action = "➕ Increase" if (stress_ch < 5 and risk_avg < 4) else (
                 "➖ Reduce"   if (stress_ch > 15 or risk_avg > 6) else "⏸ Maintain")
        shift_data.append({"Region": region, "Baseline Volume": base_vol,
                            "Avg Cost Δ%": stress_ch, "Avg Risk": risk_avg, "Action": action})
    shift_df = pd.DataFrame(shift_data).sort_values("Avg Cost Δ%")
    st.dataframe(
        shift_df.style
            .format({"Baseline Volume": "{:,}", "Avg Cost Δ%": "{:+.1f}%", "Avg Risk": "{:.2f}"}),
        use_container_width=True,
        height=280,
    )


# ── TAB: About ────────────────────────────────────────────────────────────
def tab_about():
    st.markdown("## ℹ️ About This App")
    st.markdown("""
This **Supply Chain Stress-Tester** is a production-grade analytics tool for modelling the
financial and operational impact of geopolitical, tariff, and logistics disruptions across
a global supplier network.

---

### 🏗️ Architecture

| Layer | Technology |
|---|---|
| UI Framework | Streamlit 1.x |
| Data Engine | Pandas + NumPy |
| Visualisations | Plotly (Express + Graph Objects) |
| Simulation | Monte Carlo (NumPy RNG, 1k–5k runs) |
| Data | Synthetic — 800 suppliers, seed-reproducible |

---

### 🔬 Methodology

**Tariff Shock** — User-defined tariff % replaces regional baseline; incremental cost
propagated through `BaseCost × (1 + Tariff%) × Volume`.

**Port Closure** — Affected suppliers incur a 25% rerouting surcharge and +10-day delay
penalty, compounding with the global delay multiplier.

**Delay Multiplier** — Scales all lead times; used to model pandemic/congestion scenarios.

**Monte Carlo** — 1,000 runs with ±15% Gaussian noise on stressed total cost;
outputs distribution, mean, and P95 tail risk.

**Risk Score** — Baseline 0–10 score (Beta-distributed) stressed upwards for port-closed
suppliers (+2.0) and lead-time spikes (proportional).

---

### 📦 Dataset Schema

| Column | Description |
|---|---|
| SupplierID | Unique identifier |
| Country / Region | Geographic grouping |
| Port | Primary export port |
| Category | Product category |
| Volume | Annual units |
| BaseCost | Cost per unit (USD) |
| TariffPct | Current effective tariff |
| LeadTime | Transit days (baseline) |
| RiskScore | Composite risk (0–10) |

---

### 🚀 Extensions (Roadmap)

1. **Live tariff API feed** — Integrate USITC / WTO tariff schedule REST APIs for real baseline data.
2. **Supplier substitution optimizer** — LP/MIP model that auto-suggests lowest-cost reshoring plan under constraints.
3. **Excel / PDF export** — One-click executive report generation with charts embedded.

---
*Built with Streamlit · Plotly · Pandas · NumPy*
    """)


# ============================================================================
# MAIN
# ============================================================================
def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='font-family:Space Mono;font-size:2rem;'>"
        "🔗 Supply Chain <span style='color:#f0a500'>Stress-Tester</span></h1>"
        "<p style='color:#6b7590;margin-top:-0.5rem;font-size:0.9rem;'>"
        "Model tariff shocks · port closures · geopolitical disruptions · Monte Carlo risk</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Data ─────────────────────────────────────────────────────────────────
    with st.spinner("Generating supplier dataset…"):
        df_base = generate_suppliers()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    tariff_delta, closed_ports, delay_mult, mc_runs = render_sidebar(df_base)

    # ── Simulation ────────────────────────────────────────────────────────────
    with st.spinner("Running simulation…"):
        sim = simulate(df_base, tariff_delta, closed_ports, delay_mult, mc_runs)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "⚡ Simulation", "💡 Recommendations", "ℹ️ About"]
    )

    with tab1:
        tab_overview(df_base)
    with tab2:
        tab_simulation(df_base, sim, mc_runs)
    with tab3:
        tab_recommendations(df_base, sim)
    with tab4:
        tab_about()


if __name__ == "__main__":
    main()
