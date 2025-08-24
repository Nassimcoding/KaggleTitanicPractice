from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier
import pandas as pd

import modelvoting as mv


trainfilename = "CleanTrainFileName.csv"
testfilename = "CleanTestFileName.csv"

# 讀取資料
train_df = pd.read_csv("train.csv")
train_df = mv.Process_Data(train_df,trainfilename)

random_value = 20
max_value = 500

y = train_df["Survived"]
X = train_df.drop(columns=["Survived"])

# 切分訓練集與驗證集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.12, random_state=random_value
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

param_grid = {
    'n_estimators': [100, 120, 150, 180, 200, 250, 300],
    'max_depth': [5, 8, 10, 12, None],
    'min_samples_split': [5,6,7,8,9,10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'max_features': ['sqrt', 'log2', None]
}

search = RandomizedSearchCV(
    rf, param_grid, n_iter=150, cv=3, scoring='accuracy', n_jobs=-1, random_state=42
)

search.fit(X_train, y_train)
print("Best params:", search.best_params_)
print("Best score:", search.best_score_)





