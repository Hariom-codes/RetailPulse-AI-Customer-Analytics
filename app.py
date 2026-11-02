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

st.subheader("Customer Segmentation")

customers = load_data("customer_features_churn.csv")

if customers is not None and "Customer_Segment" in customers.columns:
    st.bar_chart(customers["Customer_Segment"].value_counts())
else:
    st.info("Customer segmentation data unavailable.")

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
