import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 載入資料
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)


bins = [0, 18, 30, 40, 50, 65, 80, 120]  # 120 設一個上限，保險
labels = ['0-18', '19-30', '31-40', '41-50', '51-65', '66-80', '80+']

df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

plt.figure(figsize=(10,6))
sns.barplot(x='AgeGroup', y='Survived', data=df)
plt.title('Survival Rate by Age Group and Gender')
plt.show()