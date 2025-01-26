import catboost
from catboost import * 
import pandas as pd

from sklearn.model_selection import train_test_split

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split

import catboost as cb
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('https://biconsult.ru/img/datascience-ml-ai/titanic.csv')

# Преобразуем его в DataFrame для дальнейшей работы
titanic_df = pd.DataFrame(data)

# Выводим первые несколько строк датасета
print(titanic_df.head())

print(titanic_df.isnull().sum()) # проаверка на наличие пропусков


# МОДИФИКАЦИЯ ДАТАЕТА
# Выбираем необходимые колонки
titanic_subset = titanic_df[['Pclass', 'Sex', 'Age']].copy()

# Определяем категории по возрасту
def age_category(age):
    if age < 18:
        return 'Child'
    elif age < 60:
        return 'Adult'
    else:
        return 'Senior'

# Применяем функцию к столбцу Age
titanic_subset['AgeCategory'] = titanic_subset['Age'].apply(age_category)

# Обрабатываем пропуски в поле Age
titanic_subset['Age'].fillna(titanic_subset['Age'].mean(), inplace=True)

# Сохраняем модифицированный датасет
titanic_subset.to_csv('modified_titanic.csv', index=False)

# Выводим первые несколько строк модифицированного датасета
print(titanic_subset.head())