import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

#Just using sklearn for scaling
from sklearn.preprocessing import MinMaxScaler
#Can be done without sklearn by doing X = (X - x_min) / (x_max - x_min) and y = (y - y_min) / (y_max - y_min) for scaling

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

df = pd.read_csv('data_daily.csv')

df['Date'] = pd.to_datetime(df['# Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df = df[df['Year'] == 2021]

seq_size = 12 
X = []
y = []
for i in range(len(df) - seq_size):
    X.append(df['Receipt_Count'].values[i:i + seq_size])
    y.append(df['Receipt_Count'].values[i + seq_size])
X = np.array(X)
y = np.array(y)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y.reshape(-1, 1))

X_train, y_train = X[:int(0.8 * len(X))], y[:int(0.8 * len(X))]
X_test, y_test = X[int(0.8 * len(X)):], y[int(0.8 * len(X)):]

x_model = load_model('best_model (1).h5')

st.title('Fetch MLE Task')
mse = x_model.evaluate(X_test, y_test)



dates_2022 = pd.date_range(start='2022-01-01', end='2022-12-31')

predictions_2022 = pd.DataFrame(index=dates_2022, columns=['Predicted_Receipt_Count'])

last_observed_values = X_test[-1]

for date in dates_2022:
    input_data = np.array([last_observed_values])
    input_data = x_scaler.transform(input_data)
    predicted_count = x_model.predict(input_data)
    predicted_count = predicted_count.reshape(-1,12)
    predicted_count = y_scaler.inverse_transform(predicted_count)
    predictions_2022.loc[date] = predicted_count[0][0]
    last_observed_values = np.roll(last_observed_values, shift=-1)
    last_observed_values[-1] = predicted_count[0][0]

predictions_2022['Date'] = pd.to_datetime(predictions_2022.index)
predictions_2022['Month'] = predictions_2022['Date'].dt.month
predictions_2022.drop('Date', axis=1, inplace=True)
predictions_2022 = predictions_2022.groupby('Month').sum()

st.sidebar.subheader("Fetch MLE Intern Task")
sidebar_selection = st.sidebar.radio("Select View", ["Input Data", "Input Data Graph", "Model Performance", "Predicted Values Table", "Predicted Graph"])

monthss = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
if sidebar_selection == "Input Data":
    st.write("Input Data:")
    st.write(df)
elif sidebar_selection == 'Input Data Graph':
    new_df = df
    new_df.drop(['# Date', 'Date', 'Year'], axis=1, inplace=True)
    st.write("Receipt_Count Graph for 2021:")
    plt.plot(monthss, new_df.groupby('Month').sum(), label='Receipt_Count', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Receipt_Count')
    plt.title('Receipt_Count for 2021')
    plt.legend()
    st.pyplot(plt)
elif sidebar_selection == 'Model Performance':
    st.write(f'Mean Squared Error (MSE): {mse}')
elif sidebar_selection == "Predicted Values Table":
    st.subheader('Predictions for 2022')
    st.write("Predicted Values Table:")
    st.write(predictions_2022)
else:
    st.write("Predicted Receipt_Count Graph for 2022:")
    plt.figure(figsize=(12, 6))
    plt.plot(monthss, predictions_2022['Predicted_Receipt_Count'].groupby('Month').sum(), label='Predicted Receipt_Count', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Receipt_Count')
    plt.title('Predicted Receipt_Count for 2022')
    plt.legend()
    st.pyplot(plt)
