import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# Загрузка данных
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# Базовые модели
base_models = [
    ('lr', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)),
    ('svm', SVC(probability=True, random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42))
]
meta_model = LogisticRegression(random_state=42)

# Стекинг
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5, stack_method='predict_proba')
stacking_clf.fit(X_train, y_train)

# Оценка
y_pred = stacking_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

# Логирование в MLflow
with mlflow.start_run() as run:
    mlflow.log_param("model_type", "StackingClassifier")
    mlflow.log_param("base_models", ["LogisticRegression", "SVC", "GradientBoosting"])
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1", f1)
    mlflow.sklearn.log_model(stacking_clf, "model")
    run_id = run.info.run_id

# Сохраняем локально (опционально)
joblib.dump(stacking_clf, 'stacking_model.pkl')

# URI для развертывания
model_uri = f"runs:/{run_id}/model"
print(model_uri)
with open('best_model.txt', 'w') as f:
    f.write(model_uri)