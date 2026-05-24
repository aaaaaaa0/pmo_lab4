import os
import pandas as pd
import numpy as np
import joblib
import mlflow
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score, f1_score, accuracy_score, log_loss
from scipy.stats.mstats import winsorize
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# 1. Предобработка (как в ноутбуке)
# ---------------------------
def preprocess_titanic(df, target_col='Survived', test_size=0.2, random_state=42):
    data = df.copy()
    data.drop(columns=['PassengerId', 'Ticket', 'Cabin'], inplace=True, errors='ignore')
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Age'] = data.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
    data['Fare'] = winsorize(data['Fare'].to_numpy(), limits=(0, 0.01))

    # Title
    data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Rare', 'Ms': 'Rare', 'Mme': 'Rare', 'Don': 'Rare',
        'Lady': 'Rare', 'Sir': 'Rare', 'Capt': 'Rare', 'Countess': 'Rare',
        'Jonkheer': 'Rare'
    }
    data['Title'] = data['Title'].map(title_mapping).fillna('Rare')
    data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 19, 60, 100],
                              labels=['Child', 'Teen', 'Adult', 'Senior'])
    data['MinorAge'] = (data['Age'] < 18).astype(int)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    data['FareGroup'] = pd.qcut(data['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'], duplicates='drop')
    data.drop(columns=['Name'], inplace=True, errors='ignore')

    categorical_features = ['Embarked', 'Title', 'AgeGroup', 'FareGroup']
    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    bool_cols = data.select_dtypes(include='bool').columns
    data[bool_cols] = data[bool_cols].astype(int)

    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state, stratify=y)
    numeric_features = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch', 'Pclass']
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
    feature_names = X_train_scaled.columns.tolist()

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

# ---------------------------
# 2. Обучение стекинга
# ---------------------------
def train_stacking(X_train, y_train):
    base_models = [
        ('lr', LogisticRegression(solver='liblinear', random_state=42,
                                  class_weight='balanced', max_iter=500)),
        ('svm', SVC(kernel='rbf', gamma='scale', probability=True, random_state=42)),
        ('gb', GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000,
                                          max_depth=10, random_state=42))
    ]
    meta_model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    stacking = StackingClassifier(estimators=base_models, final_estimator=meta_model,
                                  cv=5, stack_method='predict_proba', n_jobs=-1)
    stacking.fit(X_train, y_train)
    return stacking

# ---------------------------
# 3. Главная функция
# ---------------------------
def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Загрузка данных
    df = pd.read_csv('data/titanic.csv')
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_titanic(df)

    # Обучение модели
    model = train_stacking(X_train_scaled, y_train)

    # Сохранение модели и препроцессора
    joblib.dump(model, 'models/model.pkl')
    preprocessor = {
        'scaler': scaler,
        'feature_names': feature_names
    }
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    # Оценка и логирование в MLflow
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logloss = log_loss(y_test, y_proba)

    mlflow.set_tracking_uri("file:./mlruns")   # логи в папку mlruns
    mlflow.set_experiment("titanic_stacking")
    with mlflow.start_run():
        mlflow.log_param("model_type", "StackingClassifier")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("log_loss", logloss)
        mlflow.sklearn.log_model(model, "stacking_model")

    # Создание best_model.txt (для Jenkins deploy)
    with open('best_model.txt', 'w') as f:
        f.write('models/model.pkl')

    print(f"Model saved. Test accuracy: {acc:.4f}, recall: {rec:.4f}")

if __name__ == '__main__':
    main()