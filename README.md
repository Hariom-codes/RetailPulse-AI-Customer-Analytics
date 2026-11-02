# RetailPulse – AI-Powered Customer Analytics & Demand Forecasting

RetailPulse is an end-to-end retail analytics project designed to transform retail transaction data into actionable business insights using Python, Machine Learning concepts, demand forecasting, inventory planning, and Power BI.

The project explores customer purchasing behaviour, customer segmentation, sales performance, product demand, churn patterns, and inventory replenishment.

## Project Objectives

- Analyze retail sales and profitability.
- Segment customers using RFM analysis.
- Explore customer churn patterns.
- Forecast product-level demand.
- Identify inventory replenishment requirements.
- Build an interactive 12-page Power BI dashboard.

## Technology Stack

| Area | Tools |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Forecasting | Prophet, Croston-SBA, Moving Average |
| Business Intelligence | Power BI, DAX |
| Environment | Google Colab, Jupyter Notebook |

## Dataset Overview

The project uses a custom-generated synthetic retail dataset containing six CSV files:

- `customers.csv`
- `products.csv`
- `sales_transactions.csv`
- `customer_features_churn.csv`
- `inventory_snapshot.csv`
- `daily_product_demand.csv`

Dataset overview:

- 1,500 customers
- 120 products
- 30,000 sales transactions
- 25,453 daily product-demand records
- Historical period: January 2024 – December 2025

**Note:** The dataset is synthetic and created for analytical and educational purposes. It does not represent actual transactions from a real retailer.

## Key Business Results

| KPI | Result |
|---|---:|
| Total Revenue | ₹9.55 Crore |
| Total Profit | ₹2.93 Crore |
| Total Orders | 30,000 |
| Total Quantity Sold | 52,435 |
| Average Order Value | ₹3,184.20 |
| Overall Profit Margin | 30.64% |

## Project Modules

### 1. Data Cleaning & Exploratory Data Analysis
- Inspected data types, missing values, and duplicate records.
- Converted date columns into datetime format.
- Analyzed revenue, profit, orders, and sales trends.
- Performed category-wise and monthly analysis.

### 2. Customer Segmentation – RFM Analysis
Applied Recency, Frequency, and Monetary analysis to segment 1,500 customers.

Customer segments include:
- Champions
- Loyal Customers
- Potential Loyalists
- At Risk
- Hibernating
- New / Recent Customers
- Needs Attention

The largest segment was At Risk, with 331 customers.

### 3. Customer Churn Analysis
- Examined customer churn distribution.
- Evaluated an XGBoost baseline and class-weighted model.
- Identified class imbalance and weak predictive performance.
- Avoided using churn-defining recency variables as predictive features.

**Important:** The available churn label is defined using more than 90 days of inactivity. The current analysis is therefore a churn-pattern study, not a validated future churn prediction model.

### 4. Demand Forecasting
Compared three forecasting approaches:

- Moving Average
- Croston-SBA
- Prophet

Used rolling validation across 120 products.

Overall average WAPE:

| Model | WAPE |
|---|---:|
| Croston-SBA | 156.71% |
| Moving Average | 157.63% |
| Prophet | 158.44% |

Croston-SBA performed best overall by a small margin.

The project target of 12% MAPE was not achieved. Sparse demand and many zero-sales days make forecasting challenging.

### 5. Inventory Optimization
Integrated demand forecasts with current inventory, supplier lead times, and safety stock.

Key results:

- Products requiring reorder: 12
- Recommended purchase quantity: 268 units
- Estimated purchase cost: ₹3,23,173.72

These are preliminary recommendations based on the synthetic dataset and a simplified replenishment policy.

### 6. Power BI Dashboard

Created a 12-page interactive Power BI dashboard:

1. Executive Summary
2. Sales Trends
3. Product Performance
4. Customer Analytics
5. Region & Category Analysis
6. Monthly & Seasonal Analysis
7. Customer Behaviour Analysis
8. Revenue & Profitability
9. Inventory Risk Analysis
10. Orders & Transactions Analysis
11. Advanced Analytics
12. Interactive Dashboard

The dashboard uses DAX measures, KPI cards, charts, slicers, and date-based analysis.

## Repository Structure

```text
RetailPulse-AI-Customer-Analytics/
│
├── Dataset/
│   ├── customers.csv
│   ├── products.csv
│   ├── sales_transactions.csv
│   ├── customer_features_churn.csv
│   ├── inventory_snapshot.csv
│   └── daily_product_demand.csv
│
├── Notebooks/
│   └── RetailPulse_Analytics.ipynb
│
├── PowerBI_Dashboard/
│   └── RetailPulse_AI_Powered_Customer_Analytics_Dashboard.pbix
│
├── Reports/
│   └── RetailPulse_Project_Report.pdf
│
├── Documentation/
│   └── RetailPulse_Dataset_Dictionary.xlsx
│
└── README.md
```

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/Hariom-codes/RetailPulse-AI-Customer-Analytics.git
```

2. Navigate to the project directory:

```bash
cd RetailPulse-AI-Customer-Analytics
```

3. Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost prophet openpyxl
```

4. Open the notebook:

```bash
jupyter notebook
```

5. Run `RetailPulse_Analytics.ipynb` to explore the analysis.

6. Open the `.pbix` file using Microsoft Power BI Desktop to explore the dashboard.

## Limitations & Future Improvements

- Improve forecasting accuracy for intermittent demand.
- Build a properly time-based future churn prediction dataset.
- Integrate real retail transaction data.
- Develop a Streamlit application.
- Add MLflow experiment tracking and model monitoring.
- Implement automated data pipelines and deployment.

## Project Disclaimer

This is an educational portfolio project developed using synthetic retail data. Forecasting and inventory results are experimental and should not be treated as production-ready business predictions.

---

**Developed by Hariom Upadhyay**

GitHub: [Hariom-codes](https://github.com/Hariom-codes)
