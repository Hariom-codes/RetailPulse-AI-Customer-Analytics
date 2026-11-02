from prometheus_client import Counter, Gauge, CollectorRegistry, start_http_server
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
import numpy as np
import streamlit as st
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# PROMETHEUS MONITORING SETUP
# --------------------------------------------------

@st.cache_resource
def init_prometheus():

    registry = CollectorRegistry()

    metrics = {
        "dashboard_runs": Counter(
            "retailpulse_dashboard_runs",
            "Total number of dashboard runs",
            registry=registry
        ),
        "total_revenue": Gauge(
            "retailpulse_total_revenue",
            "Total sales revenue",
            registry=registry
        ),
        "total_orders": Gauge(
            "retailpulse_total_orders",
            "Total number of orders",
            registry=registry
        ),
        "low_stock": Gauge(
            "retailpulse_low_stock_products",
            "Number of low stock products",
            registry=registry
        ),
        "high_risk": Gauge(
            "retailpulse_high_stockout_risk",
            "Number of high stockout risk products",
            registry=registry
        )
    }

    start_http_server(8001, registry=registry)

    return metrics


prometheus_metrics = init_prometheus()
prometheus_metrics["dashboard_runs"].inc()




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
def load_csv(filename):
    file_path = DATA / filename

    if not file_path.exists():
        st.error(f"Dataset not found: {filename}")
        return None

    return pd.read_csv(file_path)


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


# Update Prometheus business metrics
prometheus_metrics["total_revenue"].set(float(revenue))
prometheus_metrics["total_orders"].set(int(orders))

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
# INVENTORY OPTIMIZATION
# --------------------------------------------------

st.markdown("---")
st.header("📦 Inventory Optimization & Reorder Planning")

inventory_df = load_csv("inventory_snapshot.csv")
demand_inventory_df = load_csv("daily_product_demand.csv")

if inventory_df is not None and demand_inventory_df is not None:

    inventory_df["Product_ID"] = inventory_df["Product_ID"].astype(str)
    demand_inventory_df["Product_ID"] = demand_inventory_df["Product_ID"].astype(str)

    demand_inventory_df["Order_Date"] = pd.to_datetime(
        demand_inventory_df["Order_Date"], errors="coerce"
    )

    demand_inventory_df["Units_Sold"] = pd.to_numeric(
        demand_inventory_df["Units_Sold"], errors="coerce"
    ).fillna(0)

    # Recent 90-day average daily demand for each product
    latest_date = demand_inventory_df["Order_Date"].max()
    recent_start = latest_date - pd.Timedelta(days=89)

    recent_demand = demand_inventory_df[
        demand_inventory_df["Order_Date"].between(recent_start, latest_date)
    ]

    avg_daily_demand = (
        recent_demand.groupby("Product_ID")["Units_Sold"]
        .sum()
        .div(90)
        .rename("Avg_Daily_Demand")
    )

    inventory_df = inventory_df.join(
        avg_daily_demand, on="Product_ID"
    )

    inventory_df["Avg_Daily_Demand"] = (
        inventory_df["Avg_Daily_Demand"].fillna(0)
    )

    # Forecast-informed baseline inventory calculation
    inventory_df["Lead_Time_Demand"] = (
        inventory_df["Avg_Daily_Demand"]
        * inventory_df["Lead_Time_Days"]
    )

    inventory_df["Calculated_Reorder_Point"] = (
        inventory_df["Lead_Time_Demand"]
        + inventory_df["Safety_Stock"]
    )

    inventory_df["Recommended_Order_Qty"] = np.ceil(
        (
            inventory_df["Calculated_Reorder_Point"]
            - inventory_df["Stock_Level"]
        ).clip(lower=0)
    )

    inventory_df["Estimated_Order_Cost"] = (
        inventory_df["Recommended_Order_Qty"]
        * inventory_df["Unit_Cost"]
    )

    inventory_df["Inventory_Status"] = np.where(
        inventory_df["Recommended_Order_Qty"] > 0,
        "Reorder Required",
        "Stock Sufficient"
    )

    # KPI calculations
    reorder_items = inventory_df[
        inventory_df["Recommended_Order_Qty"] > 0
    ]

    total_products = len(inventory_df)
    reorder_count = len(reorder_items)
    total_reorder_units = int(
        inventory_df["Recommended_Order_Qty"].sum()
    )
    total_purchase_cost = inventory_df["Estimated_Order_Cost"].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Products", total_products)
    col2.metric("Products to Reorder", reorder_count)
    col3.metric("Recommended Units", total_reorder_units)
    col4.metric("Estimated Purchase Cost", f"₹{total_purchase_cost:,.0f}")

    st.markdown("### Reorder Recommendations")

    st.dataframe(
        reorder_items[
            [
                "Product_ID",
                "Category",
                "Stock_Level",
                "Lead_Time_Days",
                "Safety_Stock",
                "Calculated_Reorder_Point",
                "Recommended_Order_Qty",
                "Unit_Cost",
                "Estimated_Order_Cost"
            ]
        ].sort_values(
            "Recommended_Order_Qty", ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Inventory Status Distribution")

    status_counts = inventory_df["Inventory_Status"].value_counts()

    st.bar_chart(status_counts)

    st.markdown("### Category-wise Reorder Cost")

    category_cost = (
        reorder_items.groupby("Category")["Estimated_Order_Cost"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(category_cost)

    # Download recommendations
    csv_data = inventory_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Inventory Recommendations",
        data=csv_data,
        file_name="retailpulse_inventory_recommendations.csv",
        mime="text/csv"
    )

    st.info(
        "Planning note: Demand is estimated using the recent 90-day "
        "average daily sales rate. Reorder point = estimated lead-time "
        "demand + safety stock. These are planning estimates, not "
        "validated production purchasing recommendations."
    )



# --------------------------------------------------
# TRANSACTION RECORDS
# --------------------------------------------------

st.divider()

with st.expander("View Transaction Records"):

    st.dataframe(
        df.head(500),
        use_container_width=True
    )


# =========================================================
# DEMAND FORECASTING - PROPHET + LSTM
# =========================================================

st.header("Demand Forecasting - Prophet vs LSTM")
st.write(
    "Select a product to generate a 30-day demand forecast "
    "using Prophet and LSTM models."
)

demand_df = load_csv("daily_product_demand.csv")

if demand_df is not None:
    demand_df["Order_Date"] = pd.to_datetime(
        demand_df["Order_Date"], errors="coerce"
    )
    demand_df["Units_Sold"] = pd.to_numeric(
        demand_df["Units_Sold"], errors="coerce"
    )

    demand_df = demand_df.dropna(
        subset=["Order_Date", "Product_ID", "Units_Sold"]
    )

    product_totals = (
        demand_df.groupby("Product_ID")["Units_Sold"]
        .sum()
        .sort_values(ascending=False)
    )

    selected_product = st.selectbox(
        "Select Product ID",
        product_totals.index.tolist(),
        key="forecast_product"
    )

    product_data = demand_df[
        demand_df["Product_ID"] == selected_product
    ]

    daily_series = (
        product_data.groupby("Order_Date")["Units_Sold"]
        .sum()
        .sort_index()
    )

    daily_series = daily_series.asfreq("D", fill_value=0)

    st.write(f"Historical daily records: {len(daily_series)}")

    if len(daily_series) < 90:
        st.warning("At least 90 days of demand history is required.")

    elif st.button("Generate 30-Day Forecast", key="run_forecast"):

        import numpy as np

        with st.spinner("Training Prophet and LSTM models..."):

            forecast_results = {}
            metric_results = {}

            # ---------------- PROPHET ----------------
            try:
                from prophet import Prophet

                prophet_data = pd.DataFrame({
                    "ds": daily_series.index,
                    "y": daily_series.values
                })

                train_prophet = prophet_data.iloc[:-30].copy()
                test_prophet = prophet_data.iloc[-30:].copy()

                prophet_model_test = Prophet(
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    daily_seasonality=False
                )

                prophet_model_test.fit(train_prophet)

                test_future = prophet_model_test.make_future_dataframe(
                    periods=30,
                    freq="D"
                )

                test_prediction = prophet_model_test.predict(
                    test_future
                )["yhat"].tail(30).clip(lower=0).values

                actual = test_prophet["y"].values

                prophet_wape = (
                    np.sum(np.abs(actual - test_prediction))
                    / max(np.sum(actual), 1)
                ) * 100

                prophet_model = Prophet(
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    daily_seasonality=False
                )

                prophet_model.fit(prophet_data)

                future = prophet_model.make_future_dataframe(
                    periods=30,
                    freq="D"
                )

                prophet_forecast = (
                    prophet_model.predict(future)
                    .tail(30)["yhat"]
                    .clip(lower=0)
                    .values
                )

                forecast_results["Prophet"] = prophet_forecast
                metric_results["Prophet"] = prophet_wape

            except Exception as e:
                st.error(f"Prophet error: {e}")

            # ---------------- LSTM ----------------
            try:
                import torch
                import torch.nn as nn
                from sklearn.preprocessing import MinMaxScaler

                torch.manual_seed(42)

                class DemandLSTM(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.lstm = nn.LSTM(
                            input_size=1,
                            hidden_size=32,
                            batch_first=True
                        )
                        self.fc = nn.Linear(32, 1)

                    def forward(self, x):
                        output, _ = self.lstm(x)
                        return self.fc(output[:, -1, :])

                def train_and_predict(values, horizon=30, epochs=12):
                    values = np.asarray(values, dtype=float)

                    scaler = MinMaxScaler()
                    scaled = scaler.fit_transform(
                        values.reshape(-1, 1)
                    )

                    window = 30
                    X, y = [], []

                    for i in range(window, len(scaled)):
                        X.append(scaled[i-window:i])
                        y.append(scaled[i])

                    X = torch.tensor(
                        np.array(X), dtype=torch.float32
                    )
                    y = torch.tensor(
                        np.array(y), dtype=torch.float32
                    )

                    model = DemandLSTM()
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=0.005
                    )
                    loss_fn = nn.MSELoss()

                    model.train()

                    for _ in range(epochs):
                        optimizer.zero_grad()
                        output = model(X)
                        loss = loss_fn(output, y)
                        loss.backward()
                        optimizer.step()

                    model.eval()

                    sequence = scaled[-window:].reshape(1, window, 1)
                    predictions = []

                    with torch.no_grad():
                        for _ in range(horizon):
                            x_input = torch.tensor(
                                sequence, dtype=torch.float32
                            )
                            next_value = model(x_input).item()
                            predictions.append(next_value)

                            sequence = np.concatenate(
                                [
                                    sequence[:, 1:, :],
                                    np.array(next_value).reshape(1, 1, 1)
                                ],
                                axis=1
                            )

                    predictions = scaler.inverse_transform(
                        np.array(predictions).reshape(-1, 1)
                    ).flatten()

                    return np.maximum(predictions, 0)

                train_values = daily_series.values[:-30]
                test_values = daily_series.values[-30:]

                lstm_test_prediction = train_and_predict(
                    train_values, horizon=30
                )

                lstm_wape = (
                    np.sum(
                        np.abs(test_values - lstm_test_prediction)
                    ) / max(np.sum(test_values), 1)
                ) * 100

                lstm_future = train_and_predict(
                    daily_series.values, horizon=30
                )

                forecast_results["LSTM"] = lstm_future
                metric_results["LSTM"] = lstm_wape

            except Exception as e:
                st.error(f"LSTM error: {e}")

        # ---------------- RESULTS ----------------
        if forecast_results:

            st.subheader("30-Day Forecast")

            forecast_dates = pd.date_range(
                start=daily_series.index.max() + pd.Timedelta(days=1),
                periods=30,
                freq="D"
            )

            forecast_table = pd.DataFrame(
                forecast_results,
                index=forecast_dates
            )

            forecast_table.index.name = "Forecast_Date"

            st.line_chart(forecast_table)

            st.subheader("Model Performance - Holdout WAPE")

            metric_df = pd.DataFrame(
                list(metric_results.items()),
                columns=["Model", "WAPE (%)"]
            )

            st.dataframe(
                metric_df.style.format({"WAPE (%)": "{:.2f}%"}),
                use_container_width=True
            )

            st.subheader("Forecast Data")
            st.dataframe(
                forecast_table.round(2),
                use_container_width=True
            )

            csv_data = forecast_table.to_csv().encode("utf-8")

            st.download_button(
                "Download Forecast CSV",
                data=csv_data,
                file_name="retailpulse_30_day_forecast.csv",
                mime="text/csv"
            )

            st.info(
                "Forecasts are experimental estimates. "
                "Validate model performance before making inventory decisions."
            )

# --------------------------------------------------
# REAL-TIME METRICS & ALERTS
# --------------------------------------------------

st.markdown("---")
st.header("🚨 Business Metrics & Alerts")

alert_inventory = load_csv("inventory_snapshot.csv")
alert_sales = load_csv("sales_transactions.csv")

if alert_inventory is not None and alert_sales is not None:

    # Inventory alerts
    alert_inventory["Stock_Level"] = pd.to_numeric(
        alert_inventory["Stock_Level"], errors="coerce"
    ).fillna(0)

    alert_inventory["Reorder_Level"] = pd.to_numeric(
        alert_inventory["Reorder_Level"], errors="coerce"
    ).fillna(0)

    low_stock = alert_inventory[
        alert_inventory["Stock_Level"] <= alert_inventory["Reorder_Level"]
    ]

    high_risk = alert_inventory[
        alert_inventory["Stockout_Risk"].astype(str).str.lower() == "high"
    ]

    # Update Prometheus inventory alert metrics
    # Update Prometheus inventory alert metrics
    prometheus_metrics["low_stock"].set(int(len(low_stock)))
    prometheus_metrics["high_risk"].set(int(len(high_risk)))
    # Sales metrics
    revenue_candidates = [
        "Revenue", "Total_Revenue", "Sales",
        "Sales_Amount", "Total_Amount", "Amount"
    ]

    profit_candidates = [
        "Profit", "Total_Profit", "Profit_Amount", "Net_Profit"
    ]

    revenue_col = next(
        (c for c in revenue_candidates if c in alert_sales.columns), None
    )

    profit_col = next(
        (c for c in profit_candidates if c in alert_sales.columns), None
    )

    total_revenue = (
        pd.to_numeric(alert_sales[revenue_col], errors="coerce").sum()
        if revenue_col else 0
    )

    total_profit = (
        pd.to_numeric(alert_sales[profit_col], errors="coerce").sum()
        if profit_col else 0
    )

    profit_margin = (
        (total_profit / total_revenue) * 100
        if total_revenue > 0 else 0
    )

        # Update Prometheus inventory metrics
    prometheus_metrics["low_stock"].set(int(len(low_stock)))
    prometheus_metrics["high_risk"].set(int(len(high_risk)))

    # Alert KPIs
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Low Stock Products", len(low_stock))
    c2.metric("High Stockout Risk", len(high_risk))
    c3.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    c4.metric("Profit Margin", f"{profit_margin:.2f}%")

    st.markdown("### System Alerts")

    if len(high_risk) > 0:
        st.error(
            f"🔴 High Risk: {len(high_risk)} products have high stockout risk."
        )
    else:
        st.success("No products currently marked as high stockout risk.")

    if len(low_stock) > 0:
        st.warning(
            f"🟠 Low Stock: {len(low_stock)} products are at or below "
            "their reorder level."
        )
    else:
        st.success("Inventory levels are above the reorder thresholds.")

    if profit_margin < 15:
        st.warning(
            f"🟠 Profit margin alert: Current margin is {profit_margin:.2f}%."
        )
    else:
        st.success(
            f"Profit margin check passed: {profit_margin:.2f}%."
        )

    st.markdown("### Products Requiring Attention")

    if not low_stock.empty:
        st.dataframe(
            low_stock[
                [
                    "Product_ID",
                    "Category",
                    "Stock_Level",
                    "Reorder_Level",
                    "Stockout_Risk"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No products currently require low-stock attention.")

    st.caption(
        "Alerts are calculated from the loaded project datasets whenever "
        "the dashboard reruns. They are not connected to a live retail feed."
    )


# --------------------------------------------------
# INTERACTIVE WHAT-IF ANALYSIS
# --------------------------------------------------

st.markdown("---")
st.header("🧪 Interactive What-if Business Analysis")

whatif_sales = load_csv("sales_transactions.csv")

if whatif_sales is not None:

    # Detect revenue and profit columns automatically
    revenue_candidates = [
        "Revenue", "Total_Revenue", "Sales",
        "Sales_Amount", "Total_Amount", "Amount"
    ]

    profit_candidates = [
        "Profit", "Total_Profit",
        "Profit_Amount", "Net_Profit"
    ]

    revenue_col = next(
        (c for c in revenue_candidates if c in whatif_sales.columns),
        None
    )

    profit_col = next(
        (c for c in profit_candidates if c in whatif_sales.columns),
        None
    )

    if revenue_col is None or profit_col is None:
        st.error(
            "Revenue or Profit column not detected. "
            "Please check the sales_transactions.csv column names."
        )
        st.write("Available columns:", whatif_sales.columns.tolist())

    else:
        whatif_sales[revenue_col] = pd.to_numeric(
            whatif_sales[revenue_col], errors="coerce"
        ).fillna(0)

        whatif_sales[profit_col] = pd.to_numeric(
            whatif_sales[profit_col], errors="coerce"
        ).fillna(0)

        base_revenue = whatif_sales[revenue_col].sum()
        base_profit = whatif_sales[profit_col].sum()
        base_cost = base_revenue - base_profit

        st.markdown(
            "Adjust sales volume and selling price to explore "
            "possible business scenarios."
        )

        col1, col2 = st.columns(2)

        with col1:
            volume_change = st.slider(
                "Expected Sales Volume Change (%)",
                min_value=-50,
                max_value=100,
                value=10,
                step=5
            )

        with col2:
            price_change = st.slider(
                "Selling Price Change (%)",
                min_value=-20,
                max_value=20,
                value=5,
                step=1
            )

        volume_factor = 1 + volume_change / 100
        price_factor = 1 + price_change / 100

        projected_revenue = (
            base_revenue * volume_factor * price_factor
        )

        projected_cost = base_cost * volume_factor
        projected_profit = projected_revenue - projected_cost

        projected_margin = (
            projected_profit / projected_revenue * 100
            if projected_revenue else 0
        )

        revenue_difference = projected_revenue - base_revenue
        profit_difference = projected_profit - base_profit

        st.markdown("### Scenario Results")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Projected Revenue",
            f"₹{projected_revenue:,.0f}",
            f"₹{revenue_difference:,.0f}"
        )

        c2.metric(
            "Projected Profit",
            f"₹{projected_profit:,.0f}",
            f"₹{profit_difference:,.0f}"
        )

        c3.metric(
            "Projected Profit Margin",
            f"{projected_margin:.2f}%"
        )

        comparison_df = pd.DataFrame({
            "Metric": ["Revenue", "Profit"],
            "Current": [base_revenue, base_profit],
            "Projected": [projected_revenue, projected_profit]
        })

        st.markdown("### Current vs Projected Performance")
        st.bar_chart(
            comparison_df.set_index("Metric")
        )

        # Exportable scenario report
        scenario_report = pd.DataFrame({
            "Metric": [
                "Sales Volume Change (%)",
                "Selling Price Change (%)",
                "Current Revenue",
                "Projected Revenue",
                "Current Profit",
                "Projected Profit",
                "Revenue Difference",
                "Profit Difference",
                "Projected Profit Margin (%)"
            ],
            "Value": [
                volume_change,
                price_change,
                base_revenue,
                projected_revenue,
                base_profit,
                projected_profit,
                revenue_difference,
                profit_difference,
                projected_margin
            ]
        })

        st.download_button(
            label="📥 Download What-if Scenario Report",
            data=scenario_report.to_csv(index=False).encode("utf-8"),
            file_name="RetailPulse_WhatIf_Report.csv",
            mime="text/csv"
        )

        
        # Generate PDF report
        pdf_buffer = BytesIO()

        pdf_doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()

        story = [
            Paragraph(
                "RetailPulse - What-if Scenario Report",
                styles["Title"]
            ),
            Spacer(1, 16),
            Paragraph(
                "Interactive Business Scenario Analysis",
                styles["Heading2"]
            ),
            Spacer(1, 12)
        ]

        # Prepare formatted report table
        pdf_table_data = [["Metric", "Value"]]

        for _, row in scenario_report.iterrows():
            metric = str(row["Metric"])
            value = row["Value"]

            if "(%)" in metric:
                formatted_value = f"{float(value):,.2f}%"
            elif any(word in metric for word in [
                "Revenue", "Profit", "Difference"
            ]):
                formatted_value = f"₹{float(value):,.2f}"
            else:
                formatted_value = str(value)

            pdf_table_data.append([metric, formatted_value])

        report_table = Table(
            pdf_table_data,
            colWidths=[300, 170]
        )

        report_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B55")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ]))

        story.append(report_table)
        story.append(Spacer(1, 18))

        story.append(
            Paragraph(
                "<b>Assumptions:</b> Unit costs are assumed constant. "
                "Projected results are scenario simulations based on "
                "the selected sales-volume and selling-price changes. "
                "They are not guaranteed forecasts.",
                styles["BodyText"]
            )
        )

        pdf_doc.build(story)
        pdf_buffer.seek(0)

        st.download_button(
            label="📄 Download What-if PDF Report",
            data=pdf_buffer.getvalue(),
            file_name="RetailPulse_WhatIf_Report.pdf",
            mime="application/pdf"
        )


        st.info(
            "Scenario assumption: unit costs remain unchanged, "
            "while sales volume and selling price change according "
            "to the selected percentages. This is a simulation, "
            "not a guaranteed forecast."
        )



# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.caption(
    "Note: This demonstration uses synthetic data. "
    "Churn analysis represents inactivity patterns, "
    "not validated future churn prediction."
)
