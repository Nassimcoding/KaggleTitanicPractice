import pandas as pd

# 載入資料
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)


# 查看前幾筆資料
print(df.head(10))

# 基本統計
print(df.describe())


# 1. 新增 AgeMissed 欄位（1 = 有缺失, 0 = 無缺失）
df["AgeMissed"] = df["Age"].isna().astype(int)

# 2. 補上平均值（或你也可以選中位數）
df["Age"] = df["Age"].fillna(df["Age"].mean())

# 3. 新增 FormatAge 欄位：將 Age 做 Min-Max 標準化
df["FormatAge"] = (df["Age"] - df["Age"].min()) / (df["Age"].max() - df["Age"].min())

# 顯示結果
print(df["AgeMissed"].head(10))
print(df["FormatAge"].head(10))
print(df["Age"].head(10))


