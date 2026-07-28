import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


df = pd.read_csv("Unemployment in India.csv")

df.columns = df.columns.str.strip()

print("First 5 Rows:\n")
print(df.head())

print("\nDataset Information:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nStatistical Summary:\n")
print(df.describe())


df.dropna(inplace=True)

df['Date'] = pd.to_datetime(df['Date'])

print("\nDataset Shape:", df.shape)


plt.figure(figsize=(12,6))

plt.plot(
    df['Date'],
    df['Estimated Unemployment Rate (%)'],
    color='blue',
    linewidth=2
)

plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)

plt.show()


region = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean()

region = region.sort_values()

plt.figure(figsize=(12,8))

region.plot(
    kind='barh',
    color='orange'
)

plt.title("Average Unemployment Rate by Region")
plt.xlabel("Average Unemployment Rate (%)")

plt.show()

plt.figure(figsize=(8,6))

sns.boxplot(
    x='Area',
    y='Estimated Unemployment Rate (%)',
    data=df
)

plt.title("Urban vs Rural Unemployment")

plt.show()

plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include=np.number)

corr = numeric_df.corr()

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

before_covid = df[df['Date'] < '2020-03-01']
after_covid = df[df['Date'] >= '2020-03-01']

before_avg = before_covid['Estimated Unemployment Rate (%)'].mean()
after_avg = after_covid['Estimated Unemployment Rate (%)'].mean()

print("\nAverage Unemployment Before Covid :", before_avg)
print("Average Unemployment After Covid  :", after_avg)

plt.figure(figsize=(6,5))

plt.bar(
    ['Before Covid', 'After Covid'],
    [before_avg, after_avg]
)

plt.ylabel("Average Unemployment Rate (%)")
plt.title("Impact of COVID-19 on Unemployment")

plt.show()

df['Month'] = df['Date'].dt.month_name()

month_order = [
    'January','February','March','April','May','June',
    'July','August','September','October','November','December'
]

monthly = df.groupby('Month')['Estimated Unemployment Rate (%)'].mean()

monthly = monthly.reindex(month_order)

plt.figure(figsize=(12,6))

monthly.plot(
    marker='o',
    linewidth=2
)

plt.title("Monthly Average Unemployment Rate")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)

plt.show()

highest = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean()

highest = highest.sort_values(ascending=False)

print("\nTop 10 Regions with Highest Unemployment:\n")
print(highest.head(10))

plt.figure(figsize=(10,6))

highest.head(10).plot(
    kind='bar',
    color='red'
)

plt.title("Top 10 Regions with Highest Unemployment")
plt.xlabel("Region")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)

plt.show()

print("\n=========== PROJECT INSIGHTS ===========")
print("1. COVID-19 caused a significant increase in unemployment.")
print("2. Some regions consistently have higher unemployment rates.")
print("3. Urban areas generally experienced more unemployment during lockdown.")
print("4. Monthly trends indicate seasonal variations.")
print("5. Data visualization helps policymakers understand employment patterns.")
print("======================================")