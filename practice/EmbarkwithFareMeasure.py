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




# 計算每個 FareGroup 下各 Embarked 的比例
embark_dist = train.groupby('FareGroup')['Embarked'].value_counts(normalize=True).unstack()
embark_dist.plot(kind='bar', stacked=True)
plt.ylabel('Proportion')
plt.title('Embarked Distribution by Fare Group')
plt.show()

