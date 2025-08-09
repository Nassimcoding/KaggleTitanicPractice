import pandas as pd
import numpy as np

# Load the data and fill missing 'Fare' values
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Define quantiles and labels for binning
quantiles = [0, 0.2, 0.8, 1]
fare_labels = ['Low_Fare', 'Mid_Fare', 'High_Fare']

# Create the 'FareGroup' column
df['FareGroup'] = pd.qcut(df['Fare'], q=quantiles, labels=fare_labels, duplicates='drop')

# Group by the 'FareGroup' and calculate the sum of fares for each group
fare_sum_by_group = df.groupby('FareGroup')['Fare'].sum().reset_index()

print("各票價區間的票價總和：")
print(fare_sum_by_group)
print("-" * 30)

# Calculate the total fare for the top 20%
top_20_fare_sum = fare_sum_by_group[fare_sum_by_group['FareGroup'] == 'High_Fare']['Fare'].iloc[0]

# Calculate the total fare for the bottom 80%
bottom_80_fare_sum = fare_sum_by_group[fare_sum_by_group['FareGroup'] != 'High_Fare']['Fare'].sum()

print(f"最高 20% 票價總和: ${top_20_fare_sum:.2f}")
print(f"剩下 80% 票價總和: ${bottom_80_fare_sum:.2f}")

# Check if the 80/20 rule holds (top 20% is 4 times the rest)
is_pareto_principle_applied = top_20_fare_sum >= bottom_80_fare_sum * 4

if is_pareto_principle_applied:
    print("\n驗證結果：最高 20% 的票價總和是剩下 80% 的 4 倍或更多，符合 80/20 法則！")
else:
    fare_ratio = top_20_fare_sum / bottom_80_fare_sum
    print(f"\n驗證結果：最高 20% 的票價總和大約是剩下 80% 的 {fare_ratio:.2f} 倍，不符合 80/20 法則。")