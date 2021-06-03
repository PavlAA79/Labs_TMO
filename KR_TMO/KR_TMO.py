import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt


def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('C:\Py\Lab6_TMO\data\Life Expectancy Data.csv')
    return data

st.sidebar.header('Модели машинного обучения')
n_neighbors = st.sidebar.slider("n_neighbors for KNeighborsRegressor", 5, 50, value=5, step = 5)

st.header('Данные')
status = st.text('Загрузка данных ...')
data = load_data()
status.text('Загрузка данных завершена')

# Удаление колонок со слишком большим количеством пропусков
data.drop(['Hepatitis_B','GDP','Population',], axis = 1, inplace = True)

# Обработка пропусков 
imp_num = SimpleImputer(strategy='median')
imp_num2 = SimpleImputer(strategy='most_frequent')
data[['Life_expectancy']] = imp_num2.fit_transform(data[['Life_expectancy']])
data[['Adult_Mortality']] = imp_num.fit_transform(data[['Adult_Mortality']])
data[['Alcohol']] = imp_num.fit_transform(data[['Alcohol']])
data[['BMI']] = imp_num.fit_transform(data[['BMI']])
data[['Polio']] = imp_num.fit_transform(data[['Polio']])
data[['Total_expenditure']] = imp_num2.fit_transform(data[['Total_expenditure']])
data[['Diphtheria']] = imp_num.fit_transform(data[['Diphtheria']])
data[['thinness_1-19_years']] = imp_num.fit_transform(data[['thinness_1-19_years']])
data[['thinness_5-9_years']] = imp_num.fit_transform(data[['thinness_5-9_years']])
data[['Income_composition_of_resources']] = imp_num.fit_transform(data[['Income_composition_of_resources']])
data[['Schooling']] = imp_num2.fit_transform(data[['Schooling']])

if st.checkbox('Показать корреляционную матрицу'):
    fig1, ax = plt.subplots(figsize=(15,7))
    sns.heatmap(data.corr(), annot=True,cmap='YlGnBu', fmt='.2f')
    st.pyplot(fig1)

data_len = data.shape[0]
st.write('Количество колонок в наборе данных - {}'.format(data.shape[1]))
st.write('Количество строк в наборе данных - {}'.format(data_len))

# Кодирование категориальных признаков целочисленными значениями
le = LabelEncoder()
le.fit(data.Status) 
data.Status = le.transform(data.Status)
le.fit(data.Country) 
data.Country = le.transform(data.Country)

st.subheader('Первые пять значений:')
st.write(data.head())

scale_cols = ['Life_expectancy', 'Adult_Mortality', 'infant_deaths', 'percentage_expenditure', 'Measles',
              'BMI','under-five_deaths','Polio','Total_expenditure','Diphtheria','thinness_1-19_years',
             'thinness_5-9_years','Schooling']

if st.checkbox('Масштабирование'):
    sc1 = MinMaxScaler()
    data[scale_cols] = sc1.fit_transform(data[scale_cols])
    st.write(data.head())

st.header('Оценка качества моделей')


def preprocess_data(data):
    regr_cols = ['percentage_expenditure','BMI','Diphtheria',
        'Polio', 'Total_expenditure','Schooling']
    X = data[regr_cols]
    Y = data.Life_expectancy
    X_train,  X_test,  Y_train,  Y_test = train_test_split(X,  Y, random_state = 0, test_size = 0.1)
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = preprocess_data(data)

LR = LinearRegression()
LR.fit(X_train, Y_train)

KNN = KNeighborsRegressor(n_neighbors=n_neighbors)
KNN.fit(X_train, Y_train)

SVR = SVR()
SVR.fit(X_train, Y_train)

Tree = DecisionTreeRegressor()
Tree.fit(X_train, Y_train)

RF = RandomForestRegressor()
RF.fit(X_train, Y_train)

GB = GradientBoostingRegressor()
GB.fit(X_train, Y_train)

metrics = [r2_score, mean_absolute_error, mean_squared_error, median_absolute_error]

models_list = [LR, KNN, SVR, Tree, RF, GB]
model_names = [i.__class__.__name__ for i in models_list]
models = st.sidebar.multiselect("Выберите модели", model_names)

current_models_list = []
for i in models:
    for j in models_list:
        if i == j.__class__.__name__:
            current_models_list.append(j)

if st.sidebar.checkbox('Выполнить оценку качества'):
    for name in metrics:
        st.subheader(name.__name__)

        array_labels = []
        array_metric = []

        for func in current_models_list:
            Y_pred = func.predict(X_test)

            array_labels.append(func.__class__.__name__)
            array_metric.append(name(Y_pred, Y_test))

            st.text("{} - {}".format(func.__class__.__name__, name(Y_pred, Y_test)))

        fig, ax1 = plt.subplots(figsize=(3,3))
        pos = np.arange(len(array_metric))
        rects = ax1.barh(
            pos,
            array_metric,
            align="center",
            height=0.5,
            tick_label=array_labels,
        )
        for a, b in zip(pos, array_metric):
            plt.text(0, a - 0.1, str(round(b, 3)), color="white")
        st.pyplot(fig)