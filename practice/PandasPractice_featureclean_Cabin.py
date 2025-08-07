import pandas as pd
import numpy as np

# 載入資料
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)


# 有艙房為 1，沒有艙房為 0
df["has_cabin"] = df["Cabin"].notna().astype(int)
print(df["has_cabin"].head(10))
print(df["Cabin"].head(10))

# 假設你已經新增了 "has_cabin" 欄位
# df["has_cabin"] = df["Cabin"].notna().astype(int)
df["Cabin"] = df["Cabin"].replace("", np.nan).replace(" ", np.nan)

survival_rates_by_group = df.groupby(["Sex", "has_cabin"])["Survived"].mean()

print(survival_rates_by_group)

# 刪除 NaN 值後，印出 Cabin 欄位的所有獨特值
# print(df["Cabin"].dropna().unique())
