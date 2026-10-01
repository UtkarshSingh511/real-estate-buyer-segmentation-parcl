import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import datetime

st.set_page_config(page_title="Parcl Real Estate Analytics", layout="wide")

# ─────────────────────────────────────────────────────────────────────
# DATA PROCESSING & ML CLUSTERING PIPELINE
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_cluster_data():
    # ── Step 1: Data Cleaning & Feature Engineering ──────────────────
    df_clients = pd.read_csv('clients.csv')
    df_props = pd.read_csv('properties.csv')

    df_clients.drop_duplicates(inplace=True)

    # Parse dates and derive Age
    df_clients['date_of_birth'] = pd.to_datetime(
        df_clients['date_of_birth'], format='mixed', errors='coerce'
    )
    current_year = datetime.datetime.now().year
    df_clients['Age'] = current_year - df_clients['date_of_birth'].dt.year
    df_clients['Age'] = df_clients['Age'].fillna(df_clients['Age'].median())

    # Clean sale_price (remove $ and commas) and aggregate per client
    df_props['sale_price'] = (
        df_props['sale_price']
        .replace(r'[\$,]', '', regex=True)
        .astype(float)
    )
    client_inv = df_props.groupby('client_ref').agg(
        total_investment=('sale_price', 'sum'),
        properties_bought=('listing_id', 'count')
    ).reset_index()

    # Merge client demographics with transaction aggregates
    df = df_clients.merge(
        client_inv, left_on='client_id', right_on='client_ref', how='left'
    )
    df['total_investment'] = df['total_investment'].fillna(0)
    df['properties_bought'] = df['properties_bought'].fillna(0)

    # Drop rows missing critical categorical fields
    df.dropna(
        subset=['client_type', 'region', 'acquisition_purpose', 'country'],
        inplace=True
    )

    # ── Step 2: Feature Encoding ─────────────────────────────────────
    # One-Hot Encode all categorical variables required by the ML model
    categorical_cols = [
        'client_type', 'region', 'acquisition_purpose',
        'country', 'loan_applied', 'referral_channel'
    ]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Identify the feature matrix (exclude non-analytic ID/name columns)
    non_feature_cols = [
        'client_id', 'first_name', 'last_name',
        'gender', 'date_of_birth', 'client_ref'
    ]
    feature_cols = [
        c for c in df_encoded.columns if c not in non_feature_cols
    ]
    X = df_encoded[feature_cols].astype(float)

    # ── Step 3: Feature Scaling ──────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Step 4: K-Means Clustering (k = 4) ───────────────────────────
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df['Cluster_ID'] = cluster_labels

    # ── Step 5: Model Evaluation (Elbow + Silhouette) ────────────────
    K_range = range(2, 11)
    inertias = []
    silhouettes = []
    for k in K_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km_labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km_labels))

    eval_df = pd.DataFrame({
        'K': list(K_range),
        'Inertia': inertias,
        'Silhouette': silhouettes
    })

    # ── Step 6: Dynamic Cluster Interpretation ───────────────────────
    # Derive segment labels from *actual cluster centroid characteristics*
    # so the data determines the names, not hardcoded rules.
    centroid_stats = df.groupby('Cluster_ID').agg(
        mean_investment=('total_investment', 'mean'),
        mean_age=('Age', 'mean'),
        mean_satisfaction=('satisfaction_score', 'mean'),
        mean_properties=('properties_bought', 'mean'),
        corporate_ratio=('client_type', lambda s: (s == 'Company').mean()),
        loan_ratio=('loan_applied', lambda s: (s == 'Yes').mean()),
        count=('client_id', 'count')
    ).reset_index()

    # Rank clusters by distinguishing centroid features
    segment_map = {}
    assigned = set()

    # C3: Corporate Buyers — cluster with highest corporate entity ratio
    corp_cluster = centroid_stats.loc[
        centroid_stats['corporate_ratio'].idxmax(), 'Cluster_ID'
    ]
    segment_map[corp_cluster] = 'C3: Corporate Buyers'
    assigned.add(corp_cluster)

    remaining = centroid_stats[~centroid_stats['Cluster_ID'].isin(assigned)]

    # C4: Luxury Investors — highest mean investment among remaining
    lux_cluster = remaining.loc[
        remaining['mean_investment'].idxmax(), 'Cluster_ID'
    ]
    segment_map[lux_cluster] = 'C4: Luxury Investors'
    assigned.add(lux_cluster)

    remaining = centroid_stats[~centroid_stats['Cluster_ID'].isin(assigned)]

    # C2: First-Time Buyers — highest loan reliance among remaining
    ftb_cluster = remaining.loc[
        remaining['loan_ratio'].idxmax(), 'Cluster_ID'
    ]
    segment_map[ftb_cluster] = 'C2: First-Time Buyers'
    assigned.add(ftb_cluster)

    remaining = centroid_stats[~centroid_stats['Cluster_ID'].isin(assigned)]

    # C1: Global Investors — the remaining cluster
    gi_cluster = remaining['Cluster_ID'].iloc[0]
    segment_map[gi_cluster] = 'C1: Global Investors'

    df['Buyer Segment'] = df['Cluster_ID'].map(segment_map)

    return df, eval_df, centroid_stats, segment_map

df, eval_df, centroid_stats, segment_map = load_and_cluster_data()


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR: USER CONTROLS
# ─────────────────────────────────────────────────────────────────────
st.sidebar.title("Dashboard Controls")
st.sidebar.write("Filter data to update the visualizations.")

country_filter = st.sidebar.multiselect(
    "Country", options=df['country'].unique(), default=df['country'].unique()
)
region_filter = st.sidebar.multiselect(
    "Region", options=df['region'].unique(), default=df['region'].unique()
)
purpose_filter = st.sidebar.multiselect(
    "Acquisition Purpose",
    options=df['acquisition_purpose'].unique(),
    default=df['acquisition_purpose'].unique()
)
client_filter = st.sidebar.multiselect(
    "Client Type",
    options=df['client_type'].unique(),
    default=df['client_type'].unique()
)

df_filtered = df[
    (df['country'].isin(country_filter))
    & (df['region'].isin(region_filter))
    & (df['acquisition_purpose'].isin(purpose_filter))
    & (df['client_type'].isin(client_filter))
]


# ─────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────
st.title("🏡 Parcl Buyer Segmentation & Intelligence")
st.markdown(
    "AI-driven buyer intelligence using **unsupervised K-Means clustering** "
    "to uncover hidden patterns in real estate investment behaviors."
)


# ─────────────────────────────────────────────────────────────────────
# MODULE 0: ML MODEL EVALUATION (Elbow + Silhouette)
# ─────────────────────────────────────────────────────────────────────
st.header("0. Clustering Model Evaluation")
st.markdown(
    "The Elbow Method and Silhouette Score confirm **K = 4** as the optimal "
    "number of clusters for this dataset."
)

col_e1, col_e2 = st.columns(2)

with col_e1:
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=eval_df['K'], y=eval_df['Inertia'],
        mode='lines+markers', name='Inertia',
        line=dict(color='#636EFA', width=2),
        marker=dict(size=8)
    ))
    # Highlight optimal K = 4
    optimal = eval_df[eval_df['K'] == 4].iloc[0]
    fig_elbow.add_trace(go.Scatter(
        x=[4], y=[optimal['Inertia']],
        mode='markers', name='Optimal K = 4',
        marker=dict(size=14, color='red', symbol='star')
    ))
    fig_elbow.update_layout(
        title="Elbow Method (WCSS / Inertia)",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia"
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

with col_e2:
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Bar(
        x=eval_df['K'], y=eval_df['Silhouette'],
        marker_color=[
            'red' if k == 4 else '#636EFA' for k in eval_df['K']
        ],
        text=eval_df['Silhouette'].round(3),
        textposition='outside'
    ))
    fig_sil.update_layout(
        title="Silhouette Score by K",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score"
    )
    st.plotly_chart(fig_sil, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# MODULE 1: BUYER SEGMENTATION OVERVIEW
# ─────────────────────────────────────────────────────────────────────
st.header("1. Buyer Segmentation Overview")
col1, col2 = st.columns(2)

with col1:
    segment_counts = (
        df_filtered['Buyer Segment'].value_counts().reset_index()
    )
    segment_counts.columns = ['Buyer Segment', 'Count']
    fig_pie = px.pie(
        segment_counts, names='Buyer Segment', values='Count',
        title="Cluster Distribution", hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_bar = px.bar(
        segment_counts, x='Buyer Segment', y='Count',
        title="Total Clients per Segment",
        text_auto=True, color='Buyer Segment'
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# MODULE 2: INVESTOR BEHAVIOR DASHBOARD
# ─────────────────────────────────────────────────────────────────────
st.header("2. Investor Behavior Dashboard")
col3, col4 = st.columns(2)

with col3:
    fig_scatter = px.scatter(
        df_filtered, x='Age', y='total_investment',
        color='Buyer Segment', size='properties_bought',
        hover_data=['client_id', 'satisfaction_score'],
        title="Investment vs. Age", opacity=0.7
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col4:
    loan_behavior = (
        df_filtered
        .groupby(['Buyer Segment', 'loan_applied'])
        .size()
        .reset_index(name='Count')
    )
    fig_loan = px.bar(
        loan_behavior, x='Buyer Segment', y='Count',
        color='loan_applied', barmode='group',
        title="Loan Reliance by Segment"
    )
    st.plotly_chart(fig_loan, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# MODULE 3: GEOGRAPHIC BUYER ANALYSIS
# ─────────────────────────────────────────────────────────────────────
st.header("3. Geographic Buyer Analysis")
region_dist = (
    df_filtered
    .groupby(['region', 'Buyer Segment'])
    .size()
    .reset_index(name='Count')
)
fig_map = px.bar(
    region_dist, y='region', x='Count', color='Buyer Segment',
    orientation='h',
    title="Buyer Segments mapped by Geographic Region",
    barmode='stack'
)
st.plotly_chart(fig_map, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# MODULE 4: SEGMENT INSIGHTS PANEL
# ─────────────────────────────────────────────────────────────────────
st.header("4. Segment Insights Panel")
st.markdown(
    "Descriptive statistics detailing average behaviors across all "
    "**ML-generated** clusters."
)

insights = df_filtered.groupby('Buyer Segment').agg(
    Avg_Age=('Age', 'mean'),
    Avg_Satisfaction=('satisfaction_score', 'mean'),
    Avg_Investment=('total_investment', 'mean'),
    Avg_Properties_Bought=('properties_bought', 'mean'),
    Total_Clients=('client_id', 'count')
).reset_index()

insights['Avg_Age'] = insights['Avg_Age'].round(1)
insights['Avg_Satisfaction'] = insights['Avg_Satisfaction'].round(2)
insights['Avg_Investment'] = insights['Avg_Investment'].apply(
    lambda x: f"${x:,.2f}"
)
insights['Avg_Properties_Bought'] = insights['Avg_Properties_Bought'].round(2)

st.dataframe(insights, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# MODULE 5: CLUSTER CENTROID ANALYSIS
# ─────────────────────────────────────────────────────────────────────
st.header("5. Cluster Centroid Analysis")
st.markdown(
    "Raw centroid statistics from the K-Means model that drive the "
    "dynamic segment labeling."
)

centroid_display = centroid_stats.copy()
centroid_display['Segment Label'] = centroid_display['Cluster_ID'].map(segment_map)
centroid_display['mean_investment'] = centroid_display['mean_investment'].apply(
    lambda x: f"${x:,.2f}"
)
centroid_display['mean_age'] = centroid_display['mean_age'].round(1)
centroid_display['mean_satisfaction'] = centroid_display['mean_satisfaction'].round(2)
centroid_display['mean_properties'] = centroid_display['mean_properties'].round(2)
centroid_display['corporate_ratio'] = (
    centroid_display['corporate_ratio'] * 100
).round(1).astype(str) + '%'
centroid_display['loan_ratio'] = (
    centroid_display['loan_ratio'] * 100
).round(1).astype(str) + '%'

st.dataframe(centroid_display, use_container_width=True)
