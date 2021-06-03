import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt


def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('C:\Py\Lab6_TMO\data\states_all(1).csv',sep='	')
    return data

st.sidebar.header('Модели машинного обучения')

st.header('Данные')
status = st.text('Загрузка данных ...')
data = load_data()
status.text('Загрузка данных завершена')

type_encoder = LabelEncoder()
data["STATE"] = type_encoder.fit_transform(data["STATE"])

if st.checkbox('Показать корреляционную матрицу'):
    fig1, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    st.pyplot(fig1)

data_len = data.shape[0]
st.write('Количество строк в наборе данных - {}'.format(data_len))
st.subheader('Первые пять значений:')
st.write(data.head())

if st.checkbox('Масштабирование'):
    sc2 = MinMaxScaler()
    for col in data.columns:
        if col != 'STATE':
            data[col] = sc2.fit_transform(data[[col]])
    st.write(data.head())

st.header('Оценка качества моделей')

def preprocess_data(data):
    X = data.drop(['TOTAL_REVENUE'], axis = 1)
    Y = data.TOTAL_REVENUE
    X_train,  X_test,  Y_train,  Y_test = train_test_split(X,  Y, random_state = 0, test_size = 0.1)
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = preprocess_data(data)

LogR = LinearRegression()
LogR.fit(X_train, Y_train)

KNN_5 = KNeighborsRegressor(n_neighbors=5)
KNN_5.fit(X_train, Y_train)

BC = BaggingRegressor()
BC.fit(X_train, Y_train)

Tree = DecisionTreeRegressor()
Tree.fit(X_train, Y_train)

RF = RandomForestRegressor()
RF.fit(X_train, Y_train)

metrics = [r2_score, mean_absolute_error, mean_squared_error, median_absolute_error]

models_list = [LogR, KNN_5, BC, Tree, RF]
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