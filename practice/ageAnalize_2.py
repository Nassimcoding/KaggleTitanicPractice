import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 載入 Titanic 資料集（請根據你的檔案路徑修改）
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)  # Kaggle Titanic 的訓練資料

# 預處理欄位
df['male'] = df['Sex'].map(lambda x: 1 if x == 'male' else 0)
df['has_sibsp'] = df['SibSp'].map(lambda x: 1 if x > 0 else 0)
df['has_parch'] = df['Parch'].map(lambda x: 1 if x > 0 else 0)

# 去除 Age 為 NaN 的資料
df = df[df['Age'].notnull()]

# 年齡分組（你也可以擴展更多年齡區段）
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
        return "65 < x"
    
df['age_group'] = df['Age'].apply(age_group)

# 只保留年齡在 0-14 的資料
df = df[df['age_group'].notnull()]

# 建立多條件組合名稱
df['group'] = (
    'male=' + df['male'].astype(str) +
    ', sibsp=' + df['has_sibsp'].astype(str) +
    ', parch=' + df['has_parch'].astype(str) +
    ', age=' + df['age_group']
)

# 群組後計算生存率
grouped = df.groupby('group')['Survived'].mean().reset_index()

# 繪圖
plt.figure(figsize=(14, 6))
sns.barplot(data=grouped, x='group', y='Survived')
plt.title('Survival Rate by Gender, SibSp, Parch, and Age Group (0-14)')
plt.ylabel('Survival Rate')
plt.xlabel('Group')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.tight_layout()
plt.grid(True)
plt.show()
