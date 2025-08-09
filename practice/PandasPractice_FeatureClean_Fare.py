import pandas as pd

# 載入資料
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(f"填補後 Fare 的缺失值數量：{df['Fare'].isnull().sum()}")

df['Fare'] = df['Fare'].fillna(df.groupby('Sex')['Fare'].transform('median'))


# 顯示填補後的 Fare 缺失值數量，確認所有 NaN 都已被處理
print(f"填補後 Fare 的缺失值數量：{df['Fare'].isnull().sum()}")

# 查看填補後的資料範例
print("\n填補後的 Fare 欄位範例：")
print(df[['Sex', 'Fare']].head(10))

male_median_fare = df[df['Sex'] == 'male']['Fare'].median()
female_median_fare = df[df['Sex'] == 'female']['Fare'].median()

print(f"\n男性乘客的票價中位數：{male_median_fare:.2f}")
print(f"女性乘客的票價中位數：{female_median_fare:.2f}")

# 定義百分位數的邊界
quantiles = [0, 0.2, 0.8, 1]

# 定義分組的標籤
fare_labels = ['Low_Fare', 'Mid_Fare', 'High_Fare']

# 使用 pd.qcut() 進行分組
df['FareGroup'] = pd.qcut(df['Fare'], q=quantiles, labels=fare_labels, duplicates='drop')

# 查看新建立的 FareGroup 特徵
print("\n票價分組範例：")
print(df[['Fare', 'FareGroup']].head(10))
print("-" * 30)

# 查看每個分組的人數，確認是否符合你的百分比設定
print("各票價區間人數：")
print(df['FareGroup'].value_counts(normalize=True).mul(100))