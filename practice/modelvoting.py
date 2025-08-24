from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
import pandas as pd

def Process_Data(df,filename):

    # 查看前幾筆資料
    print(df["Pclass"].head(10))

    # 基本統計
    print(df.describe())

    # Pclass

    df["Pclass"].fillna(3, inplace=True)

    df = pd.get_dummies(df, columns=["Pclass"], prefix="Pclass")

    print(df["Pclass_1"].head(10))
    print(df["Pclass_2"].head(10))
    print(df["Pclass_3"].head(10))

    print("pclass")
    print(df.head(10))


    # Sex
    df["Sex"] = df["Sex"].fillna("male")
    gender_map = {"male": 1, "female": 0}
    df["Sex"] = df["Sex"].map(gender_map)


    print("sex")
    print(df["Sex"].head(10))


    # age
    df["AgeMissed"] = df["Age"].isna().astype(int)

    # 2. 補上平均值（或你也可以選中位數）
    df["Age"] = df["Age"].fillna(df["Age"].mean())

    # 3. 新增 FormatAge 欄位：將 Age 做 Min-Max 標準化
    df["age_norn"] = (df["Age"] - df["Age"].min()) / (df["Age"].max() - df["Age"].min())

    # 顯示結果
    print(df["AgeMissed"].head(10))
    print(df["age_norn"].head(10))
    print(df["Age"].head(10))


    # SibSp
    df['has_sibsp'] = df['SibSp'].map(lambda x: 1 if x > 0 else 0)
    df['sibsp_norm'] = (df['SibSp'] - df['SibSp'].min()) / (df['SibSp'].max() - df['SibSp'].min())

    # parch
    df['has_parch'] = df['Parch'].map(lambda x: 1 if x > 0 else 0)
    df['parch_norm'] = (df['Parch'] - df['Parch'].min()) / (df['Parch'].max() - df['Parch'].min())

    # titcket - drop



    # fare
    df['Fare'] = df.groupby('Sex')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )

    low_cutoff = df['Fare'].quantile(0.2)
    high_cutoff = df['Fare'].quantile(0.8)

    # 根據分位數標記票價分類
    def fare_level(fare):
        if fare <= low_cutoff:
            return 'low'
        elif fare >= high_cutoff:
            return 'high'
        else:
            return 'mid'

    df['fare_group'] = df['Fare'].apply(fare_level)

    print("各票價分組人數：")
    print(df['fare_group'].value_counts())

    print("票價分界線：")
    print("Low cutoff (20%):", low_cutoff)
    print("High cutoff (80%):", high_cutoff)

    df['fare_norm'] = (df['Fare'] - df['Fare'].min()) / (df['Fare'].max() - df['Fare'].min())

    fare_dummies = pd.get_dummies(df['fare_group'], prefix='fare')
    df = pd.concat([df, fare_dummies], axis=1)


    # cabin
    df['hascabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == '' else 1)


    # embark
    df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked', dummy_na=False)


    # save data
    df.to_csv('titanic_cleaned.csv', index=False)


    # drop
    df = df.drop(columns=[
        'PassengerId',
        'Name',
        'Age',
        'SibSp',
        'Parch',
        'Ticket',
        'Fare',
        'Cabin',
        'fare_group'
    ])


    # save data
    df.to_csv(filename, index=False)


    # if(df["Pclass"] == 1):
    #     df["Pclass1"] = 1
        
    # if(df["Pclass"] == 2):
    #     df["Pclass2"] = 1
        
    # if(df["Pclass"] == 3):
    #     df["Pclass3"] = 1

    # df["Pclass3"] = df["Pclass"].isna().astype(int) 
    return df

#-------------------------------------
trainfilename = "CleanTrainFileName.csv"
testfilename = "CleanTestFileName.csv"

# 讀取資料
test_df = pd.read_csv("test.csv")
test_df = Process_Data(test_df,testfilename)

random_value = 20
max_value = 500

models = {
    "Logistic Regression": LogisticRegression(max_iter=max_value),
    "Random Forest": RandomForestClassifier(random_state=random_value),
    "Gradient Boosting": GradientBoostingClassifier(random_state=random_value),
    "SVM": SVC(probability=True, random_state=random_value)
}


# 讀取資料
train_df = pd.read_csv("train.csv")
train_df = Process_Data(train_df,trainfilename)

y = train_df["Survived"]
X = train_df.drop(columns=["Survived"])

# 切分訓練集與驗證集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.05, random_state=random_value
)

# 訓練與評估
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_val, preds))
    print("-" * 50)





# 建立 VotingClassifier（軟投票）
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=max_value)),
        ('rf', RandomForestClassifier(random_state=random_value)),
        ('gb', GradientBoostingClassifier(random_state=random_value)),
        ('svc', SVC(probability=True, random_state=random_value))
    ],
    voting='soft'  # 設為 soft voting
)

# 訓練
voting_clf.fit(X_train, y_train)

# 預測與評估
voting_preds = voting_clf.predict(X_val)
voting_acc = accuracy_score(y_val, voting_preds)
print(f"Voting Classifier Accuracy: {voting_acc:.4f}")
print(classification_report(y_val, voting_preds))




# 假設 train_df 與 test_df 已經由 Process_Data 處理
train_df = pd.read_csv("CleanTrainFileName.csv")
test_df = pd.read_csv("CleanTestFileName.csv")  # 你要先把 test_df 清洗後存成這個檔案
raw_test_df = pd.read_csv("test.csv")  # 保留原始 PassengerId 方便輸出

# 拆分 X 與 y
X_train = train_df.drop(columns=["Survived"])
y_train = train_df["Survived"]

# Voting Classifier（軟投票）
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=max_value)),
        ('rf', RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=7,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1)),
        ('gb', GradientBoostingClassifier(random_state=random_value)),
        ('svc', SVC(probability=True, random_state=random_value))
    ],
    voting='soft'
)

# 訓練模型
voting_clf.fit(X_train, y_train)

# 預測 test.csv
test_preds = voting_clf.predict(test_df)

# 建立提交檔
submission = pd.DataFrame({
    "PassengerId": raw_test_df["PassengerId"],
    "Survived": test_preds
})

# 儲存成 CSV
submission.to_csv("submission.csv", index=False)
print("提交檔已輸出：submission.csv")

