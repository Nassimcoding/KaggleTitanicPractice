import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 載入資料
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
train = pd.read_csv(url)

# 建立分位數
q20 = train['Fare'].quantile(0.2)
q80 = train['Fare'].quantile(0.8)

# 分組
def fare_group(fare):
    if fare <= q20:
        return 'Low 20%'
    elif fare >= q80:
        return 'Top 20%'
    else:
        return 'Middle 60%'

train['FareGroup'] = train['Fare'].apply(fare_group)


sns.barplot(x='FareGroup', y='Survived', hue='Sex', data=train, order=['Low 20%', 'Middle 60%', 'Top 20%'])
plt.title('Survival Rate by Fare Group')
plt.show()



