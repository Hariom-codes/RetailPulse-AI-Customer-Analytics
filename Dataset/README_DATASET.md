# RetailPulse Synthetic Retail Dataset

This is a custom **synthetic** dataset created for the RetailPulse project. It is not real customer or retailer data and should be described as synthetic in the report/demo.

## Files
- customers.csv: customer profile and location dimensions (1,500 rows)
- products.csv: product catalogue and pricing/cost (120 rows)
- sales_transactions.csv: transaction-level sales (30,000 rows)
- customer_features_churn.csv: customer-level RFM-style features and churn target
- inventory_snapshot.csv: product-level inventory snapshot as of 2025-12-31
- daily_product_demand.csv: aggregated product-day demand for time-series forecasting

## Join keys
- customers.Customer_ID = sales_transactions.Customer_ID = customer_features_churn.Customer_ID
- products.Product_ID = sales_transactions.Product_ID = inventory_snapshot.Product_ID = daily_product_demand.Product_ID

## Important definitions
- Churn_Status: 1 means no purchase for more than 90 days as of 2025-12-31; 0 otherwise. This is a rule-based proxy label for learning, not observed business churn.
- Stockout_Risk: based on current Stock_Level compared with Reorder_Level.
- Revenue = Quantity * Unit_Price * (1 - Discount)
- Profit = Revenue - Quantity * Unit_Cost
- Dates cover 2024-01-01 through 2025-12-30 in transaction data.

## Suggested use
- EDA: sales_transactions.csv + customers.csv + products.csv
- Segmentation: customer_features_churn.csv
- Demand forecasting: daily_product_demand.csv (aggregate further to product-week/month if needed)
- Inventory optimization: inventory_snapshot.csv + forecast outputs

All values are generated for demonstration and portfolio practice. Validate data and avoid claiming business impact or model performance until measured.
