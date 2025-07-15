
# Cookie Cats A/B‑Test Dashboard

![cookie‑cats](https://i.ytimg.com/vi/YCo68lwTk5E/maxresdefault.jpg)

## Overview
This repository contains an end‑to‑end analysis of the **Cookie Cats** mobile‑game A/B experiment together with an interactive dashboard built in Streamlit (Power‑BI style).  
The goal is to evaluate whether moving the first “energy gate” from level 30 to level 40 improves player engagement and retention, and to provide a visual tool for product managers to explore the results.

---

## Project Goals
* **Reproduce the statistical analysis** in a Jupyter notebook (`cookie_cats.ipynb`).
* **Serve an interactive dashboard** (`dashboard/app.py`) that:
  * Displays KPI cards, histograms, boxplots and bar charts.
  * Runs on‑the‑fly statistical tests (z‑test, χ², Bayesian).
  * Includes a power‑analysis calculator for future experiments.
* Offer clear, reproducible instructions for local execution.

---

## Dataset Description
The dataset (located in `dataset/cookie_cats.csv`) contains ~90 k new players with:

| column | description |
|--------|-------------|
| `userid` | unique player ID |
| `version` | `gate_30` (control) or `gate_40` (treatment) |
| `sum_gamerounds` | rounds played on first day |
| `retention_1` | player opened the game the day after (0/1) |
| `retention_7` | player opened the game a week later (0/1) |

---

## Analysis & Dashboard Approach
### Notebook (`cookie_cats.ipynb`)
1. **Data audit**: nulls, duplicates, distribution, outliers (IQR).
2. **Exploratory plots**: histograms & log‑scale boxplots of rounds.
3. **Stat tests**  
   * Two‑proportion z‑test & χ² for retention.  
   * Bootstrap 95 % CI.  
   * Bayesian Beta‑posterior.  
4. **Power analysis**: sample size for detecting +2 pp uplift.

### Dashboard (`dashboard/app.py`)
* Built with **Streamlit**, **Plotly** and optional **streamlit‑aggrid**.
* Tabs: **Overview · EDA · Statistical Tests · Power Analysis**.
* Live filters (round‑range slider, raw‑data preview).
* KPI cards styled to mimic Power BI tiles.

---

## How to Run the Dashboard

```bash
# 1. Clone the repo
git clone https://github.com/m-zann/cookie‑cats.git
cd cookie‑cats

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/Scripts/activate        # Windows PowerShell
# or
source .venv/bin/activate            # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt      # contains streamlit, pandas, plotly...
#   OR minimal:
# pip install streamlit pandas numpy plotly scipy statsmodels streamlit-aggrid

# 4. Launch the dashboard
cd dashboard
streamlit run app.py
```

Open the provided URL (usually <http://localhost:8501>) in your browser.

> **Note**  
> If you skip installing **streamlit‑aggrid**, the app still works — tables fall back to the default Streamlit dataframe.

---

## Future Work
* Add cohort‑level drill‑downs (e.g. by install date or country).
* Deploy the dashboard on Streamlit Community Cloud.
* Integrate a REST endpoint to pull fresh data automatically.

---

## License
Released under the MIT License
