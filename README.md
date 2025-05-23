# Práctica 2 - Uso de una herramienta para desplegar un modelo de inteligencia artificial
# Autor: Guillermo Escobar Bonilla

Este proyecto corresponde a la **Práctica 2** del módulo, en la cual se debía construir un modelo de Machine Learning, registrar su ciclo de vida con MLFlow y desplegarlo mediante una aplicación web con Flask. La rúbrica exigía cumplir con criterios específicos como la gestión del dataset, uso del modelo, registro del experimento y funcionalidad web.

## Objetivos Cumplidos

| Criterio | Descripción | Estado |
|---------|-------------|--------|
| **Archivo de datos** | Uso de archivos reales (`Fake.csv`, `True.csv`) con noticias falsas y verdaderas. | Cumplido |
| **Modelo** | Entrenado con `LogisticRegression` y vectorizado con `TfidfVectorizer`. | Cumplido |
| **Ciclo de vida ML** | Usamos **MLFlow** para registrar métricas, parámetros y artefactos del modelo. | Cumplido |
| **Aplicación Web** | Servidor Flask que recibe un texto y predice si es una noticia falsa o verdadera. | Cumplido |

---

## Estructura del Proyecto

practica2_HerramientasIA/
├── dataset/
│ ├── Fake.csv
│ └── True.csv
├── model/
│ ├── model.pkl
│ └── tfidf.pkl
├── mlruns/ # Experimentos registrados por MLFlow
├── app.py # API Flask que expone el modelo
├── train_model.py # Entrenamiento y registro con MLFlow
├── requirements.txt # Librerías necesarias
└── README.md

---

## Instrucciones de uso local

### 1. Crear entorno virtual e instalar dependencias

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


### 2. Entrenar modelo

python train_model.py

### 3. Correr MLFlow en el puerto 9090

mlflow ui --backend-store-uri file:./mlruns --port 9090
Abrir en el navegador: http://localhost:9090

### 4. Ejecutar el servidor Flask

python app.py
