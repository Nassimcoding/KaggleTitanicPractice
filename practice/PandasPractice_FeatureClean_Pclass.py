import pandas as pd

# 載入資料
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)


# 查看前幾筆資料
print(df["Pclass"].head(10))

# 基本統計
print(df.describe())

# Pclass

df["Pclass"].fillna(3, inplace=True)

df = pd.get_dummies(df, columns=["Pclass"], prefix="Pclass")

print(df["Pclass_1"].head(10))
print(df["Pclass_2"].head(10))
print(df["Pclass_3"].head(10))

print("pclass")
print(df.head(10))


# Sex
df["Sex"] = df["Sex"].fillna("male")
gender_map = {"male": 1, "female": 0}
df["Sex"] = df["Sex"].map(gender_map)


print("sex")
print(df["Sex"].head(10))


# age
df["AgeMissed"] = df["Age"].isna().astype(int)

# 2. 補上平均值（或你也可以選中位數）
df["Age"] = df["Age"].fillna(df["Age"].mean())

# 3. 新增 FormatAge 欄位：將 Age 做 Min-Max 標準化
df["age_norn"] = (df["Age"] - df["Age"].min()) / (df["Age"].max() - df["Age"].min())

# 顯示結果
print(df["AgeMissed"].head(10))
print(df["age_norn"].head(10))
print(df["Age"].head(10))


# SibSp
df['has_sibsp'] = df['SibSp'].map(lambda x: 1 if x > 0 else 0)
df['sibsp_norm'] = (df['SibSp'] - df['SibSp'].min()) / (df['SibSp'].max() - df['SibSp'].min())

# parch
df['has_parch'] = df['Parch'].map(lambda x: 1 if x > 0 else 0)
df['parch_norm'] = (df['Parch'] - df['Parch'].min()) / (df['Parch'].max() - df['Parch'].min())

# titcket - drop



# fare
df['Fare'] = df.groupby('Sex')['Fare'].transform(
    lambda x: x.fillna(x.median())
)

low_cutoff = df['Fare'].quantile(0.2)
high_cutoff = df['Fare'].quantile(0.8)

# 根據分位數標記票價分類
def fare_level(fare):
    if fare <= low_cutoff:
        return 'low'
    elif fare >= high_cutoff:
        return 'high'
    else:
        return 'mid'

df['fare_group'] = df['Fare'].apply(fare_level)

print("各票價分組人數：")
print(df['fare_group'].value_counts())

print("票價分界線：")
print("Low cutoff (20%):", low_cutoff)
print("High cutoff (80%):", high_cutoff)

df['fare_norm'] = (df['Fare'] - df['Fare'].min()) / (df['Fare'].max() - df['Fare'].min())

fare_dummies = pd.get_dummies(df['fare_group'], prefix='fare')
df = pd.concat([df, fare_dummies], axis=1)


# cabin
df['hascabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == '' else 1)


# embark
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked', dummy_na=False)


# save data
df.to_csv('titanic_cleaned.csv', index=False)


# drop
df = df.drop(columns=[
    'PassengerId',
    'Name',
    'Age',
    'SibSp',
    'Parch',
    'Ticket',
    'Fare',
    'Cabin',
    'fare_group'
])


# save data
df.to_csv('titanic_ReadyToTrain.csv', index=False)


# if(df["Pclass"] == 1):
#     df["Pclass1"] = 1
    
# if(df["Pclass"] == 2):
#     df["Pclass2"] = 1
    
# if(df["Pclass"] == 3):
#     df["Pclass3"] = 1

# df["Pclass3"] = df["Pclass"].isna().astype(int) 


