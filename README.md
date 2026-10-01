# Machine Learning based Buyer Segmentation and Investment Profiling

## Project Overview
This repository contains a comprehensive data science project developed for **Parcl Co. Limited** (in collaboration with Unified Mentor). The objective is to transition real estate marketing from a generalized approach to a highly targeted, data-driven strategy. By leveraging unsupervised machine learning algorithms (K-Means Clustering), this project analyzes property transactions and client demographics to uncover hidden investment patterns and segment buyers into distinct, actionable profiles.

## Technology Stack

| Category | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.9+ | Core programming and data manipulation |
| **Data Processing** | Pandas, NumPy | Data cleaning, feature engineering, and aggregation |
| **Machine Learning** | Scikit-Learn | One-Hot Encoding, StandardScaler, K-Means Clustering |
| **Web Application** | Streamlit | Rapid deployment of the interactive analytics dashboard |
| **Data Visualization** | Plotly (Express & Graph Objects) | Interactive charts, maps, and dynamic dashboard UI |

## Project Structure

```text
real-estate-buyer-segmentation-parcl/
│
├── app.py                   # Main Streamlit application and ML pipeline
├── clients.csv              # Raw demographic dataset
├── properties.csv           # Raw property transaction dataset
├── research_paper.md        # Full research paper and methodology documentation
├── requirements.txt         # Deployment dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation (this file)
```

## Local Setup & Installation

**1. Clone the repository:**

```bash
git clone https://github.com/<your-username>/real-estate-buyer-segmentation-parcl.git
cd real-estate-buyer-segmentation-parcl
```

**2. Set up the virtual environment:**

```bash
# Standard Python venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on WSL / Linux
source venv/bin/activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Launch the Streamlit Dashboard:**

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` in your default browser.

## Data Science Methodology

The project follows a rigorous six-step machine learning pipeline:

| Step | Description | Implementation |
| :--- | :--- | :--- |
| **Step 1** | Data Cleaning & Feature Engineering | Deduplication, date parsing, median imputation, transaction aggregation |
| **Step 2** | Feature Encoding | One-Hot Encoding via `pd.get_dummies()` for all categorical variables |
| **Step 3** | Feature Scaling | `StandardScaler` normalization on the full encoded feature matrix |
| **Step 4** | K-Means Clustering | `KMeans(n_clusters=4)` unsupervised clustering on scaled features |
| **Step 5** | Model Evaluation | Elbow Method (WCSS/Inertia) + Silhouette Score validation |
| **Step 6** | Cluster Interpretation | Dynamic centroid-based segment labeling (no hardcoded rules) |

## Key Findings & Buyer Segments

The machine learning model successfully identified **four optimal buyer segments** (K=4), validated via Silhouette Score and the Elbow Method:

| Segment | Profile | Key Characteristics |
| :--- | :--- | :--- |
| **C1** | Global Investors | High total investment volume, predominantly cash/equity purchases |
| **C2** | First-Time Buyers | Younger demographic, heavily reliant on financing and loans |
| **C3** | Corporate Buyers | B2B entities acquiring multiple units with distinct geographical focus |
| **C4** | Luxury Investors | High net-worth individuals making massive investments with top satisfaction scores |

## Dashboard Modules

The Streamlit web application includes the following interactive modules:

- **Module 0 — Clustering Model Evaluation:** Elbow Method and Silhouette Score charts confirming K=4
- **Module 1 — Buyer Segmentation Overview:** Donut chart and bar chart showing cluster distribution
- **Module 2 — Investor Behavior Dashboard:** Investment vs. Age scatter plot and Loan Reliance analysis
- **Module 3 — Geographic Buyer Analysis:** Stacked horizontal bar chart mapping segments by region
- **Module 4 — Segment Insights Panel:** Descriptive statistics matrix per ML-generated cluster
- **Module 5 — Cluster Centroid Analysis:** Raw centroid statistics driving the dynamic segment labeling

## Business Recommendations

1. **Targeted Financing Campaigns (C2):** Deploy customized mortgage assistance and first-time buyer guides to increase conversion rates.
2. **Premium Concierge Services (C4):** Assign dedicated relationship managers and off-market property previews for luxury investors.
3. **Bulk Investment Offerings (C3):** Route corporate entities to specialized B2B sales teams offering multi-unit portfolio discounts.
4. **International Marketing Adjustments (C1):** Reallocate digital ad spend to international regions, highlighting yield potential and macroeconomic stability.

## License

This project was developed as part of an academic and professional engagement with Parcl Co. Limited and Unified Mentor.
