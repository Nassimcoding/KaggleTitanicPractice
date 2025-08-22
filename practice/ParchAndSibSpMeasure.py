import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 載入資料
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)




df['Alone'] = (df['SibSp'] + df['Parch'] == 0).astype(int)


sns.barplot(x='Alone', y='Survived', hue='Sex', data=df)
plt.xticks([0, 1], ['With Family', 'Alone'])
plt.title('Survival Rate: Alone vs With Family')
plt.show()
