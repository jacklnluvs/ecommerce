# E-Commerce Sales Forecasting Using Machine Learning

## Project Overview

This project predicts e-commerce sales using historical order data and machine learning. The dataset consists of customer orders, order items, products, users, reviews, and events. A Random Forest Regressor is trained to forecast item sales.

---

# Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

# Dataset Files

- orders.csv
- order_items.csv
- products.csv
- users.csv
- reviews.csv
- events.csv

---

# Step 1: Import Libraries

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

### Output

Libraries imported successfully.

---

# Step 2: Load Dataset

## Code

```python
orders = pd.read_csv("orders.csv")
order_items = pd.read_csv("order_items.csv")
products = pd.read_csv("products.csv")
users = pd.read_csv("users.csv")
reviews = pd.read_csv("reviews.csv")
events = pd.read_csv("events.csv")
```

### Output

All CSV files loaded successfully.

---

# Step 3: Display Dataset

## Code

```python
orders.head()
```

### Output

Displays the first five rows of the Orders dataset.

Example

| order_id | user_id | order_date | order_status | total_amount |
|----------|---------|------------|--------------|-------------|
| 1 | 101 | 2024-01-01 | Delivered | 2500 |

---

# Step 4: Dataset Information

## Code

```python
orders.info()
```

### Output

```
RangeIndex: xxxx entries
Columns: 5
order_id
user_id
order_date
order_status
total_amount
```

Shows:
- Number of rows
- Data types
- Missing values

---

# Step 5: Missing Values

## Code

```python
orders.isnull().sum()
```

### Output

```
order_id        0
user_id         0
order_date      0
order_status    0
total_amount    0
```

---

# Step 6: Duplicate Values

## Code

```python
orders.duplicated().sum()
```

### Output

```
0
```

---

# Step 7: Merge Dataset

## Code

```python
merged_df = pd.merge(
    orders,
    order_items,
    on='order_id',
    how='inner'
)
```

### Output

Merged dataset created successfully.

---

# Step 8: Display Merged Dataset

## Code

```python
merged_df.head()
```

### Output

Shows merged table containing order details and item details.

---

# Step 9: Convert Date

## Code

```python
merged_df['order_date'] = pd.to_datetime(merged_df['order_date'])
```

### Output

Date column converted successfully.

---

# Step 10: Feature Engineering

## Code

```python
merged_df['Year'] = merged_df['order_date'].dt.year
merged_df['Month'] = merged_df['order_date'].dt.month
merged_df['Day'] = merged_df['order_date'].dt.day
```

### Output

Three new columns created:

- Year
- Month
- Day

---

# Step 11: Select Features

## Code

```python
X = merged_df[['Year','Month','Day','quantity','item_price']]
y = merged_df['item_total']
```

### Output

Input features and target variable selected.

---

# Step 12: Train-Test Split

## Code

```python
X_train,X_test,y_train,y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)
```

### Output

80% Training Data

20% Testing Data

---

# Step 13: Train Model

## Code

```python
model = RandomForestRegressor(
n_estimators=100,
random_state=42
)

model.fit(X_train,y_train)
```

### Output

Random Forest model trained successfully.

---

# Step 14: Prediction

## Code

```python
y_pred = model.predict(X_test)
```

### Output

Predicted sales values generated.

---

# Step 15: Model Evaluation

## Code

```python
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

print(mae)
print(mse)
print(rmse)
print(r2)
```

### Output

Example

```
Mean Absolute Error : 12.45

Mean Squared Error : 520.31

Root Mean Squared Error : 22.81

R² Score : 0.94
```

*(Your values will depend on the dataset.)*

---

# Step 16: Visualization

## Code

```python
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.show()
```

### Output



<img width="1386" height="1084" alt="Screenshot 2026-07-17 181448" src="https://github.com/user-attachments/assets/3973b913-88d2-43d6-a716-67876dbc31f8" />

# Results

- Dataset successfully loaded and cleaned.
- Orders and Order Items merged successfully.
- Date-based features extracted.
- Random Forest model trained.
- Sales predicted successfully.
- Model evaluated using:
  - MAE
  - MSE
  - RMSE
  - R² Score

---

# Conclusion

The project demonstrates how machine learning can be used to forecast e-commerce sales. Historical order data was processed, engineered into useful features, and used to train a Random Forest Regressor. The model provides sales predictions and performance metrics, helping businesses estimate future sales trends and make informed decisions.
