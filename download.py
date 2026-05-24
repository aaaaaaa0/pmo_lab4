import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Загрузка данных
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Удаление ненужных колонок
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Заполнение пропусков
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Преобразование категориальных признаков
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)  # создаст Embarked_Q, Embarked_S

# (Опционально) создание новых признаков, если они были в вашей модели
# Например, FamilySize, IsAlone, Title и т.д. – добавьте по необходимости.
# Для простоты оставляем базовые признаки.

# Признаки и цель
X = df.drop('Survived', axis=1)
y = df['Survived']

# Стратифицированное разделение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Сохранение
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Данные Titanic подготовлены и сохранены.")