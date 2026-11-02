
import numpy as np
import streamlit as st
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RetailPulse")
st.subheader("AI-Powered Customer Analytics & Demand Forecasting")
st.caption("Retail analytics demonstration using a synthetic dataset.")


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

DATA = Path(__file__).parent / "Dataset"


@st.cache_data
def load_data(filename):
    path = DATA / filename

    if path.exists():
        return pd.read_csv(path)

    return None


sales = load_data("sales_transactions.csv")

if sales is None:
    st.error("Dataset/sales_transactions.csv not found.")
    st.stop()


sales["Order_Date"] = pd.to_datetime(
    sales["Order_Date"],
    errors="coerce"
)

sales = sales.dropna(subset=["Order_Date"])


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Filters")

start = sales["Order_Date"].min().date()
end = sales["Order_Date"].max().date()

dates = st.sidebar.date_input(
    "Select date range",
    value=(start, end)
)

df = sales.copy()

if isinstance(dates, (tuple, list)) and len(dates) == 2:
    df = df[
        df["Order_Date"].dt.date.between(
            dates[0], dates[1]
        )
    ]

if "Category" in df.columns:
    categories = sorted(df["Category"].dropna().unique())

    selected = st.sidebar.multiselect(
        "Select category",
        categories,
        default=categories
    )

    df = df[df["Category"].isin(selected)]


# --------------------------------------------------
# EXECUTIVE OVERVIEW
# --------------------------------------------------

revenue = df["Revenue"].sum()
profit = df["Profit"].sum()
orders = df["Order_ID"].nunique()
quantity = df["Quantity"].sum()

aov = revenue / orders if orders else 0
margin = profit / revenue * 100 if revenue else 0

st.markdown("### Executive Overview")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Revenue", f"₹{revenue:,.0f}")
c2.metric("Total Profit", f"₹{profit:,.0f}")
c3.metric("Total Orders", f"{orders:,}")
c4.metric("Average Order Value", f"₹{aov:,.0f}")
c5.metric("Profit Margin", f"{margin:.1f}%")

st.divider()


# --------------------------------------------------
# SALES ANALYSIS
# --------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("Revenue by Category")

    category_sales = df.groupby("Category")["Revenue"].sum()

    st.bar_chart(category_sales)

with right:
    st.subheader("Monthly Revenue Trend")

    monthly = df.copy()

    monthly["Month"] = (
        monthly["Order_Date"]
        .dt.to_period("M")
        .astype(str)
    )

    trend = monthly.groupby("Month")["Revenue"].sum()

    st.line_chart(trend)

st.divider()


# --------------------------------------------------
# RFM CUSTOMER SEGMENTATION
# --------------------------------------------------

st.subheader("Customer Segmentation (RFM Analysis)")

rfm = sales.groupby("Customer_ID").agg(
    Total_Orders=("Order_ID", "nunique"),
    Total_Spending=("Revenue", "sum"),
    Total_Quantity=("Quantity", "sum"),
    Last_Purchase=("Order_Date", "max")
).reset_index()

reference_date = pd.Timestamp("2025-12-31")

rfm["Recency"] = (
    reference_date - rfm["Last_Purchase"]
).dt.days


# RFM scoring

rfm["R_Score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    q=5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["F_Score"] = pd.qcut(
    rfm["Total_Orders"].rank(method="first"),
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M_Score"] = pd.qcut(
    rfm["Total_Spending"].rank(method="first"),
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)


# Original seven business segments

def segment_customer(row):

    r = row["R_Score"]
    f = row["F_Score"]
    m = row["M_Score"]

    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"

    elif r >= 3 and f >= 4 and m >= 3:
        return "Loyal Customers"

    elif r >= 4 and f == 3 and m >= 3:
        return "Potential Loyalists"

    elif r >= 4 and f <= 2:
        return "New / Recent Customers"

    elif r <= 2 and (f >= 3 or m >= 4):
        return "At Risk"

    elif r <= 2 and f <= 2 and m <= 3:
        return "Hibernating"

    else:
        return "Needs Attention"


rfm["Customer_Segment"] = rfm.apply(
    segment_customer,
    axis=1
)

st.markdown("### RFM Segment Distribution")

segment_counts = rfm["Customer_Segment"].value_counts()

st.bar_chart(segment_counts)

st.markdown("### RFM Segment Summary")

rfm_summary = rfm.groupby("Customer_Segment").agg(
    Total_Customers=("Customer_ID", "count"),
    Average_Spending=("Total_Spending", "mean"),
    Total_Revenue=("Total_Spending", "sum")
).round(2)

st.dataframe(
    rfm_summary,
    use_container_width=True
)


# --------------------------------------------------
# MACHINE LEARNING CUSTOMER SEGMENTATION
# K-MEANS + DBSCAN
# --------------------------------------------------

st.divider()

st.subheader("Machine Learning Customer Segmentation")

st.write(
    "This section applies two unsupervised machine learning "
    "algorithms to customer purchase behaviour."
)

# Select clustering features

cluster_features = rfm[
    [
        "Recency",
        "Total_Orders",
        "Total_Spending"
    ]
].copy()

# Reduce skewness in frequency and monetary values

cluster_features["Total_Orders"] = np.log1p(
    cluster_features["Total_Orders"]
)

cluster_features["Total_Spending"] = np.log1p(
    cluster_features["Total_Spending"]
)

# Handle missing or infinite values

cluster_features = cluster_features.replace(
    [np.inf, -np.inf],
    np.nan
)

valid_rows = cluster_features.notna().all(axis=1)

if valid_rows.sum() < 6:

    st.warning(
        "Not enough valid customer records for clustering."
    )

else:

    # Standardize features

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        cluster_features.loc[valid_rows]
    )

    # ----------------------------------------------
    # K-MEANS
    # ----------------------------------------------

    st.markdown("### 1. K-Means Clustering")

    kmeans = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    kmeans_labels = kmeans.fit_predict(
        scaled_features
    )

    rfm["KMeans_Cluster"] = np.nan

    rfm.loc[
        valid_rows,
        "KMeans_Cluster"
    ] = kmeans_labels

    kmeans_counts = pd.Series(
        kmeans_labels
    ).value_counts().sort_index()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Customers in Each Cluster**")
        st.bar_chart(kmeans_counts)

    with c2:

        kmeans_score = silhouette_score(
            scaled_features,
            kmeans_labels
        )

        st.metric(
            "K-Means Silhouette Score",
            f"{kmeans_score:.3f}"
        )

    kmeans_summary = rfm[
        rfm["KMeans_Cluster"].notna()
    ].groupby("KMeans_Cluster").agg(
        Customers=("Customer_ID", "nunique"),
        Average_Recency=("Recency", "mean"),
        Average_Orders=("Total_Orders", "mean"),
        Average_Spending=("Total_Spending", "mean")
    ).round(2)

    st.markdown("**K-Means Cluster Profile**")

    st.dataframe(
        kmeans_summary,
        use_container_width=True
    )

    # ----------------------------------------------
    # DBSCAN
    # ----------------------------------------------

    st.markdown("### 2. DBSCAN Clustering")

    dbscan = DBSCAN(
        eps=0.8,
        min_samples=10
    )

    dbscan_labels = dbscan.fit_predict(
        scaled_features
    )

    rfm["DBSCAN_Cluster"] = np.nan

    rfm.loc[
        valid_rows,
        "DBSCAN_Cluster"
    ] = dbscan_labels

    dbscan_counts = pd.Series(
        dbscan_labels
    ).value_counts().sort_index()

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("**DBSCAN Cluster Distribution**")

        st.bar_chart(dbscan_counts)

    with c2:

        noise_count = int(
            (dbscan_labels == -1).sum()
        )

        valid_dbscan_labels = dbscan_labels[
            dbscan_labels != -1
        ]

        unique_clusters = np.unique(
            valid_dbscan_labels
        ).size

        st.metric(
            "DBSCAN Noise Points",
            f"{noise_count:,}"
        )

        st.metric(
            "DBSCAN Clusters Found",
            f"{unique_clusters:,}"
        )

    # Calculate DBSCAN silhouette without noise

    dbscan_mask = dbscan_labels != -1

    if (
        np.unique(dbscan_labels[dbscan_mask]).size >= 2
        and dbscan_mask.sum() > unique_clusters
    ):

        dbscan_score = silhouette_score(
            scaled_features[dbscan_mask],
            dbscan_labels[dbscan_mask]
        )

        st.metric(
            "DBSCAN Silhouette Score (excluding noise)",
            f"{dbscan_score:.3f}"
        )

    else:

        st.info(
            "DBSCAN did not identify enough non-noise "
            "clusters to calculate a silhouette score."
        )

    # ----------------------------------------------
    # DBSCAN CLUSTER PROFILE
    # ----------------------------------------------

    dbscan_summary = rfm[
        rfm["DBSCAN_Cluster"].notna()
    ].groupby("DBSCAN_Cluster").agg(
        Customers=("Customer_ID", "nunique"),
        Average_Recency=("Recency", "mean"),
        Average_Orders=("Total_Orders", "mean"),
        Average_Spending=("Total_Spending", "mean")
    ).round(2)

    st.markdown("**DBSCAN Cluster Profile**")

    st.dataframe(
        dbscan_summary,
        use_container_width=True
    )

    st.caption(
        "Note: K-Means uses five predefined clusters. "
        "DBSCAN identifies density-based clusters and noise points. "
        "Cluster IDs are algorithm labels, not predefined business names. "
        "Noise points are represented by cluster label -1."
    )


# --------------------------------------------------
# CUSTOMER CHURN & INACTIVITY ANALYSIS
# --------------------------------------------------

st.divider()

st.subheader("Customer Churn & Inactivity Analysis")

churn = load_data("customer_features_churn.csv")

if churn is not None:

    churn["Churn_Status"] = pd.to_numeric(
        churn["Churn_Status"],
        errors="coerce"
    )

    churn = churn.dropna(
        subset=["Churn_Status"]
    )

    churn["Churn_Status"] = churn[
        "Churn_Status"
    ].astype(int)

    churn["Customer_Status"] = churn[
        "Churn_Status"
    ].map({
        0: "Active",
        1: "Churned"
    })

    total_customers = len(churn)

    churned_customers = (
        churn["Churn_Status"] == 1
    ).sum()

    active_customers = (
        churn["Churn_Status"] == 0
    ).sum()

    churn_rate = (
        churned_customers / total_customers * 100
        if total_customers else 0
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    c2.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    c3.metric(
        "90-Day Inactivity Rate",
        f"{churn_rate:.1f}%"
    )

    left, right = st.columns(2)

    with left:

        st.markdown("**Active vs Churned Customers**")

        status_counts = churn[
            "Customer_Status"
        ].value_counts()

        st.bar_chart(status_counts)

    with right:

        st.markdown(
            "**Days Since Last Purchase Distribution**"
        )

        inactivity = pd.to_numeric(
            churn["Days_Since_Last_Purchase"],
            errors="coerce"
        ).dropna()

        if not inactivity.empty:

            bins = pd.cut(
                inactivity,
                bins=10
            )

            distribution = (
                bins.value_counts()
                .sort_index()
            )

            distribution.index = (
                distribution.index.astype(str)
            )

            st.bar_chart(distribution)

    st.markdown("**Customer Behaviour Comparison**")

    comparison = churn.groupby(
        "Customer_Status"
    ).agg(
        Customers=("Customer_ID", "nunique"),
        Average_Orders=("Total_Orders", "mean"),
        Average_Revenue=("Total_Revenue", "mean"),
        Average_Order_Value=("Average_Order_Value", "mean"),
        Average_Tenure=("Customer_Tenure_Days", "mean")
    ).round(2).reset_index()

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.info(
        "Churn is defined as more than 90 days without a purchase "
        "as of 31 December 2025. This analysis describes historical "
        "customer inactivity, not validated future churn prediction."
    )

else:

    st.warning(
        "Customer churn dataset not found."
    )


# --------------------------------------------------
# INVENTORY OVERVIEW
# --------------------------------------------------

st.divider()

st.subheader("Inventory Overview")

inventory = load_data("inventory_snapshot.csv")

if inventory is not None:

    st.dataframe(
        inventory.head(20),
        use_container_width=True
    )

else:

    st.info("Inventory data unavailable.")


# --------------------------------------------------
# TRANSACTION RECORDS
# --------------------------------------------------

st.divider()

with st.expander("View Transaction Records"):

    st.dataframe(
        df.head(500),
        use_container_width=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.caption(
    "Note: This demonstration uses synthetic data. "
    "Churn analysis represents inactivity patterns, "
    "not validated future churn prediction."
)
