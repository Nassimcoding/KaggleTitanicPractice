import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 載入 Titanic 資料集（請根據你的檔案路徑修改）
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)  # Kaggle Titanic 的訓練資料

# 新增性別欄位（male = 1, female = 0）
df['male'] = df['Sex'].map(lambda x: 1 if x == 'male' else 0)

# 是否有兄弟姊妹/配偶（SibSp > 0）
df['has_sibsp'] = df['SibSp'].map(lambda x: 1 if x > 0 else 0)

# 是否有父母/小孩（Parch > 0）
df['has_parch'] = df['Parch'].map(lambda x: 1 if x > 0 else 0)

# 過濾出年齡在 0~14 歲之間的乘客
age_filter = df[(df['Age'] >= 0) & (df['Age'] <= 120)].copy()

# 加上年齡區間分類
def age_group(age):
    if age <= 4:
        return '0-4'
    elif age <= 9:
        return '5-9'
    elif age <= 14:
        return '10-14'
    elif age <= 19:
        return '15-19'
    elif age <= 24:
        return '20-24'
    elif age <= 29:
        return '25-29'
    elif age <= 34:
        return '30-34'
    elif age <= 39:
        return '35-39'
    elif age <= 44:
        return '40-44'
    elif age <= 49:
        return '45-49'
    elif age <= 54:
        return '50-54'
    elif age <= 59:
        return '55-59'
    elif age <= 64:
        return '60-64'
    else:
        return "over 65"
    
age_filter['age_group'] = age_filter['Age'].apply(age_group)

# 計算每個年齡區間的生存率
grouped = age_filter.groupby('age_group')['Survived'].mean().reset_index()

# 繪製圖表
plt.figure(figsize=(64, 48))
sns.barplot(data=grouped, x='age_group', y='Survived')
plt.title('Survival Rate by Age Group (0-65)')
plt.ylabel('Survival Rate')
plt.xlabel('Age Group')
plt.ylim(0, 1)
plt.grid(True)
plt.show()
