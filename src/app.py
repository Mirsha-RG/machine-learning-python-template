from utils import db_connect
import streamlit as st
import pickle
import numpy as np

engine = db_connect()

# your code here



# Cargar el modelo
with open('modelo.pkl', 'rb') as f:
    model = pickle.load(f)
print("✅ Modelo entrenado y guardado como 'modelo.pkl'")

st.title("Predicción con mi modelo de Machine Learning")

# Entradas del usuario
feature1 = st.number_input("Ingresa el valor de Pregnancies ")
feature2 = st.number_input("Ingresa el valor de Glucose ")
feature3 = st.number_input("Ingresa el valor de BloodPressure ")
feature4 = st.number_input("Ingresa el valor de SkinThickness ")
feature5 = st.number_input("Ingresa el valor de Insulin ")
feature6 = st.number_input("Ingresa el valor de BMI ")
feature7 = st.number_input("Ingresa el valor de Age ")
feature8 = st.number_input("Ingresa el valor de DiabetesPedigreeFunction ")
#feature9 = st.number_input("Ingresa el valor de Outcome ")


""" 
if st.button("Predecir"):
  
    datos = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]])  
    prediccion = model.predict(datos)
    st.write(f"Resultado de la predicción: {prediccion[0]}")

"""

if st.button("Predecir"):
    datos = np.array([[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8]]) 
    prediccion = model.predict(datos)
    resultados = {0: "No tiene la condición", 1: "Tiene la condición"}
    st.write("Resultado:", resultados.get(prediccion[0], "Resultado desconocido"))


