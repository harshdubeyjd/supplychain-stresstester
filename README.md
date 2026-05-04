# 🔗 Supply Chain Stress-Tester

A production-grade Streamlit analytics app for modelling the financial and operational
impact of geopolitical shocks, tariff escalations, and port disruptions across an
800-supplier global network.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **What-If Sidebar** | Per-region tariff sliders, multi-port closure selector, delay multiplier |
| **Preset Scenarios** | US-China trade war, Red Sea crisis, Pandemic shock, EU CBAM, Taiwan Strait |
| **Monte Carlo** | 500 – 5,000 simulation runs with P95 tail-risk output |
| **World Map** | Plotly Geo scatter — baseline & stressed risk by supplier |
| **Cost Breakdown** | Before/after grouped bar by region |
| **Risk Heatmap** | Region × Category stressed risk |
| **Tornado Chart** | Cost delta sensitivity by region |
| **Recommendations** | Sourcing shift table + top-20 risk suppliers + executive summary |

---

## 🚀 Quick Start

### 1. Clone / copy the files
```
supply_chain_app/
├── app.py
├── requirements.txt
└── README.md
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## 🌐 Deploy to Streamlit Community Cloud (free)

1. Push the folder to a public GitHub repo
2. Go to https://share.streamlit.io → "New app"
3. Point to your `app.py` — done, no extra config needed

---

## 📂 File Guide

| File | Description |
|---|---|
| `app.py` | Full application — data gen, simulation engine, all UI tabs |
| `requirements.txt` | Minimal pinned dependencies |
| `README.md` | This file |

---

## 🔬 Simulation Methodology

- **Tariff Shock** — replaces regional baseline tariff; cost = `BaseCost × (1+Tariff%) × Volume`
- **Port Closure** — 25% rerouting surcharge + 10-day delay per affected supplier
- **Delay Multiplier** — scales all lead times (pandemic / congestion proxy)
- **Monte Carlo** — ±15% Gaussian noise on stressed total cost; outputs mean and P95
- **Risk Score** — Beta(2,5) baseline, stressed +2 for port closure, +proportional for lead-time spike

---

## 🚀 Suggested Extensions

1. **Live tariff API** — Pull real USITC/WTO tariff schedules as baseline instead of synthetic defaults
2. **LP Sourcing Optimizer** — Auto-suggest lowest-cost reshoring plan using PuLP or OR-Tools under volume/lead-time constraints
3. **PDF Executive Report** — One-click export of KPIs + charts using ReportLab or WeasyPrint

---

*Stack: Streamlit · Plotly · Pandas · NumPy · Python 3.11+*
