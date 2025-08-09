import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 載入 Titanic 資料集
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 預處理欄位
df['male'] = df['Sex'].map(lambda x: 1 if x == 'male' else 0)
df['has_sibsp'] = df['SibSp'].map(lambda x: 1 if x > 0 else 0)
df['has_parch'] = df['Parch'].map(lambda x: 1 if x > 0 else 0)

# 去除 Age 為 NaN 的資料
df = df[df['Age'].notnull()]

# 年齡分組（可擴展）
def age_group(age):
    if age <= 4:
        return '0-4'
    elif age <= 9:
        return '5-9'
    elif age <= 14:
        return '10-14'
    elif age <= 19:
        return '15-19'
    elif age <= 29:
        return '20-29'
    elif age <= 39:
        return '30-39'
    elif age <= 49:
        return '40-49'
    elif age <= 64:
        return '50-64'
    else:
        return '65+'

df['age_group'] = df['Age'].apply(age_group)

# 建立繪圖的子圖架構（2 行 × 4 列）
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# 設定樣式
sns.set(style="whitegrid")

# 所有 8 種條件組合
combinations = [
    (m, s, p)
    for m in [0, 1]
    for s in [0, 1]
    for p in [0, 1]
]

# 遍歷每個條件組合，畫一張子圖
for idx, (m, s, p) in enumerate(combinations):
    ax = axes[idx]
    subset = df[
        (df['male'] == m) &
        (df['has_sibsp'] == s) &
        (df['has_parch'] == p)
    ]
    
    if subset.empty:
        ax.set_title(f"male={m}, sibsp={s}, parch={p}\n(No Data)")
        ax.axis('off')
        continue

    # 計算該組的年齡分組生存率
    grouped = subset.groupby('age_group')['Survived'].mean().reset_index()

    sns.barplot(data=grouped, x='age_group', y='Survived', ax=ax)
    ax.set_title(f"male={m}, sibsp={s}, parch={p}")
    ax.set_ylim(0, 1)
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Survival Rate')
    ax.tick_params(axis='x', rotation=45)

# 全圖標題
plt.suptitle('Survival Rate by Age Group for Each Gender + SibSp + Parch Combination', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
