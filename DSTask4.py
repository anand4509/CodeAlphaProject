# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load Dataset
df = pd.read_csv("Advertising.csv")
df.head()
# Data Cleaning
print(df.info())

print(df.isnull().sum())

df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
# Exploratory Data Analysis
sns.pairplot(df)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.show()
# Feature Selection
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
# Train Regression Model
model = LinearRegression()

model.fit(X_train, y_train)
# Predict Sales
predictions = model.predict(X_test)
# Evaluate Model
print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("R2 Score:", r2_score(y_test, predictions))
# Predict Future Sales
future = pd.DataFrame({
    'TV':[250],
    'Radio':[40],
    'Newspaper':[30]
})

future_sales = model.predict(future)

print("Predicted Sales:", future_sales)
# Analyzing Advertising impact
coeff = pd.DataFrame({
    'Feature': X.columns,
    'Impact': model.coef_
})

print(coeff)
# Visualization
plt.figure(figsize=(8,5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
