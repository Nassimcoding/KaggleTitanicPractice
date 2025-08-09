import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 讀取資料
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 性別 male = 1, female = 0
df['male'] = df['Sex'].map(lambda x: 1 if x == 'male' else 0)

# 補 `Fare` 缺失值，以性別的中位數補
df['Fare'] = df.groupby('male')['Fare'].transform(lambda x: x.fillna(x.median()))

# 分出票價區段
low_cutoff = df['Fare'].quantile(0.2)
high_cutoff = df['Fare'].quantile(0.8)

def fare_group(fare):
    if fare <= low_cutoff:
        return 'low'
    elif fare >= high_cutoff:
        return 'high'
    else:
        return 'mid'

df['fare_group'] = df['Fare'].apply(fare_group)

# 群組並計算生存率
grouped = df.groupby(['male', 'fare_group'])['Survived'].mean().reset_index()

# 將性別轉回字串標籤方便看
grouped['Sex'] = grouped['male'].map({0: 'Female', 1: 'Male'})

# 顯示結果表格
print("性別 + 票價分組 的生存率：")
print(grouped[['Sex', 'fare_group', 'Survived']])

# 繪製圖表
plt.figure(figsize=(8, 6))
sns.barplot(data=grouped, x='fare_group', y='Survived', hue='Sex')
plt.title('Survival Rate by Fare Group and Gender')
plt.ylabel('Survival Rate')
plt.xlabel('Fare Group')
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()
