
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np

# 1. Load Dataset
df = pd.read_csv("train.csv")

print("----- DATASET INFORMATION -----")
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nFirst 5 Rows:")
print(df.head())

# 2. Check Missing Values
print("\n----- MISSING VALUES -----")
print(df.isnull().sum().sort_values(ascending=False).head(15))

# 3. Separate Input Features and Output Label
# Remove Id because it is only an identification number
X = df.drop(["SalePrice", "Id"], axis=1)
y = df["SalePrice"]

print("\nInput Features Shape:", X.shape)
print("Output Label Shape:", y.shape)

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns
categorical_features = X.select_dtypes(exclude=["number"]).columns

# 4. Handle Missing Values and Convert Categorical Data
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 5. Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# 6. Train Linear Regression Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully!")

# 7. Make Predictions
y_pred = model.predict(X_test)

print("\n----- ACTUAL VS PREDICTED PRICES -----")

results = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": y_pred[:10]
})

print(results.round(2))

# 8. Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n----- MODEL EVALUATION -----")
print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# 9. Generate Graph
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.savefig("house_price_prediction.png")
plt.show()

# 10. Save Predictions
results.to_csv("predictions.csv", index=False)

print("\nHouse Price Prediction Project Completed!")
