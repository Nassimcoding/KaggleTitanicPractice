import pandas as pd

# 載入資料
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 查看前幾筆資料
print(df.head())

# 基本統計
print(df.describe())

# 篩選出女性乘客
women = df[df['Sex'] == 'female']
firstclasswomen = women[women["Pclass"] == 1] 
women_survival_rate = women['Survived'].mean()
print(f"woman total survival rate: {women_survival_rate:.2%}")

firstclasswomen_survival_rate = firstclasswomen['Survived'].mean()
print(f"first class woman total survival rate: {firstclasswomen_survival_rate:.2%}")



# 計算生還率
survival_rate = df['Survived'].mean()
print(f"整體生還率: {survival_rate:.2%}")

