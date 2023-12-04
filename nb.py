from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import streamlit as st
import pickle
import numpy as np
import pandas as pd

#Tahap Preprocessing
# Membaca dataset
data = pd.read_csv('New_Algeria.csv')

# Memisahkan fitur dan target
X = data.drop('label', axis=1)
X1 = X.drop('day', axis=1)
X2 = X1.drop('month', axis=1)
X3 = X2.drop('year', axis=1)
X4 = X3.drop('Temperature', axis=1)
X5 = X4.drop('rh', axis=1)
X6 = X5.drop('ws', axis=1)
X7 = X6.drop('rain', axis=1)
X8 = X7.drop('ffmc', axis=1)
X9 = X8.drop('dmc', axis=1)
X10 = X9.drop('dc', axis=1)
X11 = X10.drop('isi', axis=1)
X12 = X11.drop('bui', axis=1)


scaler7 = MinMaxScaler()
Fitur7 = scaler7.fit_transform(X6)

# Memisahkan label target
y = data['label']

# Pisahkan data menjadi data latih dan uji
X6_train, X6_test, y_train, y_test = train_test_split(Fitur7, y, test_size=0.2, random_state=42)



# Load model
with open('7fitur.pkl', 'rb') as file:
    nb_model = pickle.load(file)

# Load scaler
with open('7skalar.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Tampilan aplikasi Streamlit
def main():
    st.title('Aplikasi Prediksi Kebakaran Hutan')

    # Input data untuk prediksi
    st.header('Masukkan Data Untuk Prediksi')
    features = []
    for i in range(1, 8):
        feature = st.number_input(f'Fitur {i}', min_value=0.000000, max_value=1.000000)
        features.append(feature)

    # Tombol prediksi
    if st.button('Prediksi'):
        # Transformasi data input dengan MinMax scaler
        normalized_data = scaler7.transform([features])

        # Melakukan prediksi menggunakan model
        prediction = nb_model.predict(normalized_data)

        # Menampilkan hasil prediksi
        st.header('Hasil Prediksi')
        st.write(f'Prediksi: {prediction[0]}')

# Menjalankan aplikasi Streamlit
if __name__ == '__main__':
    main()
