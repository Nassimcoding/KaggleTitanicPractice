import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 載入資料
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
train = pd.read_csv(url)


sns.barplot(x='Embarked', y='Survived', hue='Sex', data=train)
plt.title('Survival Rate by Embarked')
plt.show()
