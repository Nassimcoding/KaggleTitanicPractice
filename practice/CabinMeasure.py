import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 載入資料
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
train = pd.read_csv(url)

# 有填寫 Cabin = 1, 沒填寫 = 0
train['HasCabin'] = train['Cabin'].notna().astype(int)



sns.barplot(x='HasCabin', y='Survived', hue='Sex', data=train)
plt.xticks([0,1], ['No Cabin','Has Cabin'])
plt.title('Survival Rate by Cabin Availability and Gender')
plt.show()

