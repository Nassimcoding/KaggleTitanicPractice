import pandas as pd

# 載入資料
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(df["SibSp"].head(10))

# 創建一個新的欄位，HasParentChild。
# 如果 Parch > 0，則 HasParentChild 為 1；否則為 0。
df['HasParentChild'] = (df['Parch'] > 0).astype(int)

# 查看新欄位與原始 Parch 的關係
print(df[['Parch', 'HasParentChild']].head(10))

df['HasSibSp'] = (df['SibSp'] > 0).astype(int)

# 查看新欄位與原始 SibSp 和 Parch 的關係
print(df[['SibSp', "HasSibSp"]].head(10))


survival_rate_analysis = df.groupby(['Sex', 'HasParentChild',"HasSibSp"])['Survived'].mean().reset_index()

# 為了讓結果更易讀，我們可以將 HasParentChild 的數值替換成有意義的標籤
survival_rate_analysis['HasParentChild'] = survival_rate_analysis['HasParentChild'].map({0: 'No Parents/Children', 1: 'Has Parents/Children'})

survival_rate_analysis['HasSibSp'] = survival_rate_analysis['HasSibSp'].map({0: 'No SibSp', 1: 'Has SibSp'})


# 定義年齡區間的邊界
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]

# 定義年齡區間的標籤
labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']

# 使用 pd.cut() 進行分組
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)





# 查看新建立的 AgeGroup 特徵
print("\n年齡分組範例：")
print(df[['Age', 'AgeGroup']].head(10))



# 根據 AgeGroup 進行分組，並計算生存率的平均值
agegroup_survival_analysis = df.groupby(['AgeGroup'])['Survived'].mean().reset_index()

# 顯示分析結果
print("\n不同年齡層的生存率分析結果：")
print(agegroup_survival_analysis)




# 顯示分析結果
print("\n分組生存率分析結果：")
print(survival_rate_analysis)
