import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import joblib

# Cargar datasets
df_fake = pd.read_csv("dataset/Fake.csv")
df_true = pd.read_csv("dataset/True.csv")

# Etiquetar los datos
df_fake["label"] = 1
df_true["label"] = 0
df = pd.concat([df_fake, df_true], ignore_index=True)

# Separar características y etiquetas
X = df["text"]
y = df["label"]

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorización
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Guardar el vectorizador
joblib.dump(tfidf, "model/tfidf.pkl")

# Configurar MLFlow
mlflow.set_tracking_uri("file:./mlruns")  # Carpeta local
mlflow.set_experiment("Fake News Detection")

# Entrenar y registrar con MLFlow
with mlflow.start_run():
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train_tfidf, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_tfidf))
    
    # Guardar el modelo
    joblib.dump(model, "model/model.pkl")
    
    # Registrar en MLFlow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_features", 5000)
    
    print(f"Modelo entrenado con precisión: {accuracy:.4f}")
