import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Загрузка артефактов
model = joblib.load('models/model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')
scaler = preprocessor['scaler']
feature_names = preprocessor['feature_names']

app = FastAPI()

# Модель входных данных (все поля Titanic)
class Passenger(BaseModel):
    PassengerId: int
    Pclass: int
    Name: str
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: str = None
    Embarked: str

def prepare_features(raw_data: Dict[str, Any]) -> np.ndarray:
    """Преобразует сырой JSON в масштабированный вектор признаков."""
    df = pd.DataFrame([raw_data])
    data = df.copy()

    # --- Копия логики из preprocess_titanic (без split и масштабирования числовых) ---
    data.drop(columns=['PassengerId', 'Ticket', 'Cabin'], inplace=True, errors='ignore')
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].fillna('S')
    # Возраст: если пропуск, заполняем медианой по классу (заранее вычисленной, но для простоты используем глобальную)
    # В учебных целях используем фиксированную медиану (28). В реальном проекте нужно сохранить возрастные медианы.
    data['Age'] = data['Age'].fillna(28.0)
    # Winsorize Fare (99-й перцентиль = 67.55 из тренировочных данных)
    fare_99th = 67.55
    data['Fare'] = data['Fare'].apply(lambda x: min(x, fare_99th))

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
    fare_quantiles = [7.91, 14.45, 31.0]   # квантили из тренировки
    def assign_fare_group(fare):
        if fare <= fare_quantiles[0]:
            return 'Low'
        elif fare <= fare_quantiles[1]:
            return 'Medium'
        elif fare <= fare_quantiles[2]:
            return 'High'
        else:
            return 'VeryHigh'
    data['FareGroup'] = data['Fare'].apply(assign_fare_group)
    data.drop(columns=['Name'], inplace=True, errors='ignore')

    categorical_features = ['Embarked', 'Title', 'AgeGroup', 'FareGroup']
    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    bool_cols = data.select_dtypes(include='bool').columns
    data[bool_cols] = data[bool_cols].astype(int)

    # Приводим к тем же колонкам, что и при обучении
    for col in feature_names:
        if col not in data.columns:
            data[col] = 0
    data = data[feature_names]

    # Масштабирование числовых признаков
    numeric_features = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch', 'Pclass']
    data[numeric_features] = scaler.transform(data[numeric_features])
    return data.values

@app.post("/predict")
def predict(passenger: Passenger):
    try:
        feat = prepare_features(passenger.dict())
        pred = model.predict(feat)[0]
        proba = model.predict_proba(feat)[0].tolist()
        return {"survived": int(pred), "probability": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}

# Запуск: uvicorn app:app --host 0.0.0.0 --port 5003