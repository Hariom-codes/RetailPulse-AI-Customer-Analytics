import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="RetailPulse", page_icon="📊", layout="wide")

st.title("📊 RetailPulse")
st.subheader("AI-Powered Customer Analytics & Demand Forecasting")
st.caption("Retail analytics demonstration using a synthetic dataset.")

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
    sales["Order_Date"], errors="coerce"
)

st.sidebar.header("Filters")

start = sales["Order_Date"].min().date()
end = sales["Order_Date"].max().date()

dates = st.sidebar.date_input(
    "Select date range",
    value=(start, end)
)

df = sales.copy()

if isinstance(dates, tuple) and len(dates) == 2:
    df = df[
        df["Order_Date"].dt.date.between(dates[0], dates[1])
    ]

if "Category" in df.columns:
    categories = sorted(df["Category"].dropna().unique())
    selected = st.sidebar.multiselect(
        "Select category", categories, default=categories
    )
    df = df[df["Category"].isin(selected)]

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

left, right = st.columns(2)

with left:
    st.subheader("Revenue by Category")
    category_sales = df.groupby("Category")["Revenue"].sum()
    st.bar_chart(category_sales)

with right:
    st.subheader("Monthly Revenue Trend")
    monthly = df.copy()
    monthly["Month"] = monthly["Order_Date"].dt.to_period("M").astype(str)
    trend = monthly.groupby("Month")["Revenue"].sum()
    st.line_chart(trend)

st.divider()

st.subheader("Customer Segmentation (RFM Analysis)")

# Calculate RFM from transaction data
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

# RFM scoring - same as original notebook
rfm["R_Score"] = pd.qcut(
    rfm["Recency"],
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

# Original 7-segment business rules
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
    segment_customer, axis=1
)

segment_counts = rfm["Customer_Segment"].value_counts()

st.bar_chart(segment_counts)

st.dataframe(
    rfm.groupby("Customer_Segment").agg(
        Total_Customers=("Customer_ID", "count"),
        Average_Spending=("Total_Spending", "mean"),
        Total_Revenue=("Total_Spending", "sum")
    ).round(2),
    use_container_width=True
)



st.divider()

st.subheader("Customer Churn & Inactivity Analysis")

churn = load_data("customer_features_churn.csv")

if churn is not None:

    # Convert churn labels to numeric
    churn["Churn_Status"] = pd.to_numeric(
        churn["Churn_Status"], errors="coerce"
    )

    churn = churn.dropna(subset=["Churn_Status"])
    churn["Churn_Status"] = churn["Churn_Status"].astype(int)

    # 0 = Active, 1 = Churned
    churn["Customer_Status"] = churn["Churn_Status"].map({
        0: "Active",
        1: "Churned"
    })

    total_customers = len(churn)
    churned_customers = (churn["Churn_Status"] == 1).sum()
    active_customers = (churn["Churn_Status"] == 0).sum()

    churn_rate = (
        churned_customers / total_customers * 100
        if total_customers else 0
    )

    # KPI metrics
    c1, c2, c3 = st.columns(3)

    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churned Customers", f"{churned_customers:,}")
    c3.metric("90-Day Inactivity Rate", f"{churn_rate:.1f}%")

    # Customer distribution
    left, right = st.columns(2)

    with left:
        st.markdown("**Active vs Churned Customers**")
        status_counts = churn["Customer_Status"].value_counts()
        st.bar_chart(status_counts)

    with right:
        st.markdown("**Days Since Last Purchase Distribution")

        inactivity = pd.to_numeric(
            churn["Days_Since_Last_Purchase"], errors="coerce"
        ).dropna()

        if not inactivity.empty:
            bins = pd.cut(inactivity, bins=10)
            distribution = bins.value_counts().sort_index()
            distribution.index = distribution.index.astype(str)
            st.bar_chart(distribution)

    # Customer behaviour comparison
    
   st.markdown("**Customer Behaviour Comparison**")

comparison = churn.groupby("Customer_Status").agg(
    Customers=("Customer_ID", "nunique"),
    Average_Orders=("Total_Orders", "mean"),
    Average_Revenue=("Total_Revenue", "mean"),
    Average_Order_Value=("Average_Order_Value", "mean"),
    Average_Tenure=("Customer_Tenure_Days", "mean")
).round(2).reset_index()

st.dataframe(comparison, use_container_width=True)


    st.info(
        "Churn is defined as more than 90 days without a purchase "
        "as of 31 December 2025. This analysis describes historical "
        "customer inactivity, not validated future churn prediction."
    )

else:
    st.warning("Customer churn dataset not found.")


st.subheader("Inventory Overview")

inventory = load_data("inventory_snapshot.csv")

if inventory is not None:
    st.dataframe(inventory.head(20), use_container_width=True)
else:
    st.info("Inventory data unavailable.")

with st.expander("View Transaction Records"):
    st.dataframe(df.head(500), use_container_width=True)

st.caption(
    "Note: This demonstration uses synthetic data. "
    "Churn analysis represents inactivity patterns, "
    "not validated future churn prediction."
)
