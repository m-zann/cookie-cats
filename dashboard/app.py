
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.stats.power import zt_ind_solve_power
import math

# === Optional ag‑Grid import ===
try:
    from st_aggrid import AgGrid, GridOptionsBuilder  # pip install streamlit-aggrid
    AG_GRID = True
except ModuleNotFoundError:
    AG_GRID = False
    st.warning(
        "⚠️  Optional dependency **streamlit‑aggrid** not found. "
        "Tables will fallback to Streamlit's default view.\n"
        "Install it with:\n\n"
        "`pip install streamlit-aggrid`"
    )

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Cookie Cats – Power‑BI‑Style Dashboard",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- HELPER FUNCTIONS ----------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # Trim extreme round outlier
    df = df[df['sum_gamerounds'] < df['sum_gamerounds'].max()]
    return df

def ztest(success_a, size_a, success_b, size_b):
    p1, p2 = success_a/size_a, success_b/size_b
    p_pool = (success_a + success_b)/(size_a + size_b)
    se = math.sqrt(p_pool*(1-p_pool)*(1/size_a + 1/size_b))
    z = (p1 - p2)/se
    p_value = 2*(1-stats.norm.cdf(abs(z)))
    return p1, p2, z, p_value

def bayes_prob(success_a, size_a, success_b, size_b, draws=50000, seed=0):
    rng = np.random.default_rng(seed)
    sims_a = rng.beta(success_a+0.5, size_a-success_a+0.5, draws)
    sims_b = rng.beta(success_b+0.5, size_b-success_b+0.5, draws)
    return (sims_a > sims_b).mean()

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Controls")
data_file = st.sidebar.text_input("CSV path", "../dataset/cookie_cats.csv")
df = load_data(data_file)

st.sidebar.markdown("#### Filters")
max_rounds = int(df['sum_gamerounds'].max())
rounds_range = st.sidebar.slider("Total rounds range", 0, max_rounds, (0, max_rounds))
df_filtered = df[df['sum_gamerounds'].between(*rounds_range)]

if st.sidebar.checkbox("Show raw data sample"):
    st.sidebar.dataframe(df_filtered.head())

# ---------- TOP METRIC CARDS ----------
st.markdown("## 🐱 Cookie Cats A/B Test Results")
col_a, col_b, col_c = st.columns(3)

col_a.metric("Players analysed", f"{len(df_filtered):,}")
col_b.metric("Avg. rounds", f"{df_filtered['sum_gamerounds'].mean():.1f}")
uplift = (
    df_filtered[df_filtered['version']=="gate_30"]['retention_7'].mean() -
    df_filtered[df_filtered['version']=="gate_40"]['retention_7'].mean()
)*100
col_c.metric("7‑day retention Δ (pp)", f"{uplift:+.2f}")

st.markdown("---")

# ---------- TABS ----------
tab_overview, tab_eda, tab_stats, tab_power = st.tabs(
    ["📊 Overview", "🔍 EDA", "📐 Statistical Tests", "⚡ Power Analysis"]
)

with tab_overview:
    st.subheader("Retention KPI comparison")
    kpis = df_filtered.groupby('version').agg(
        players=('userid','count'),
        avg_rounds=('sum_gamerounds','mean'),
        retention_1=('retention_1','mean'),
        retention_7=('retention_7','mean')
    ).reset_index()
    kpis[['retention_1','retention_7']] *= 100

    if AG_GRID:
        from st_aggrid import GridOptionsBuilder, AgGrid
        gb = GridOptionsBuilder.from_dataframe(kpis)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_default_column(editable=False, groupable=False)
        gridOptions = gb.build()
        AgGrid(kpis, gridOptions=gridOptions, height=220, theme='material')
    else:
        st.dataframe(kpis)

    bar = px.bar(
        kpis.melt(id_vars='version', value_vars=['retention_1','retention_7']),
        x='variable', y='value', color='version', barmode='group',
        labels={'value':'Retention %', 'variable':'Metric'},
        title='Retention by version',
        text_auto='.2f'
    )
    bar.update_layout(legend_title_text='Version', yaxis_tickformat='.1f')
    st.plotly_chart(bar, use_container_width=True)

with tab_eda:
    st.subheader("Distribution of total rounds")
    hist = px.histogram(
        df_filtered, x='sum_gamerounds', color='version', nbins=60, opacity=0.75,
        labels={'sum_gamerounds':'Total rounds'},
        title='Histogram of game rounds (filtered)'
    )
    hist.update_xaxes(range=[0, min(rounds_range[1], 2000)])
    st.plotly_chart(hist, use_container_width=True)

    st.subheader("Log‑scale boxplot of rounds")
    box = px.box(
        df_filtered, x='version', y='sum_gamerounds', color='version',
        log_y=True, points='outliers',
        labels={'sum_gamerounds':'Total rounds'},
        title='Log‑scale distribution of rounds by version'
    )
    st.plotly_chart(box, use_container_width=True)

with tab_stats:
    st.subheader("Frequentist results")
    z_results = {}
    for metric in ['retention_1','retention_7']:
        s = df_filtered.groupby('version')[metric].agg(['sum','count'])
        p1,p2,z,pv = ztest(s.loc['gate_30','sum'], s.loc['gate_30','count'],
                           s.loc['gate_40','sum'], s.loc['gate_40','count'])
        chi2, pv_chi2, *_ = chi2_contingency(pd.crosstab(df_filtered['version'], df_filtered[metric]))
        z_results[metric] = dict(
            gate_30=f"{p1:.2%}",
            gate_40=f"{p2:.2%}",
            z=f"{z:.2f}",
            p=f"{pv:.4f}",
            chi2=f"{chi2:.2f}",
            p_chi2=f"{pv_chi2:.4f}"
        )
    st.table(pd.DataFrame(z_results).T)

    st.subheader("Bayesian probabilities")
    bayes = {}
    for metric in ['retention_1','retention_7']:
        s = df_filtered.groupby('version')[metric].agg(['sum','count'])
        prob = bayes_prob(s.loc['gate_30','sum'], s.loc['gate_30','count'],
                          s.loc['gate_40','sum'], s.loc['gate_40','count'])
        bayes[metric] = {"P(gate_30 > gate_40)": f"{prob:.2%}"}
    st.table(pd.DataFrame(bayes).T)

with tab_power:
    st.subheader("Sample‑size calculator")
    baseline = df_filtered[df_filtered['version']=='gate_30']['retention_1'].mean()
    uplift_pp = st.slider("Minimum Detectable Effect (pp)", 0.5, 5.0, 2.0, step=0.5)
    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, step=0.01)
    power_target = st.slider("Power (1‑β)", 0.50, 0.95, 0.80, step=0.05)
    uplift = uplift_pp/100
    prop2 = baseline + uplift
    sd = math.sqrt(baseline*(1-baseline) + prop2*(1-prop2))
    effect_size = uplift / sd
    n_req = zt_ind_solve_power(effect_size, alpha=alpha, power=power_target)
    st.metric("Players required per group", f"{math.ceil(n_req):,}")

# ---------- FOOTER / GLOBAL CSS ----------
st.markdown("""
<style>
.stMetric {background-color:#68759d; border-radius:8px; padding:10px;}
div.block-container {padding-top:1rem;}
header, footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)
