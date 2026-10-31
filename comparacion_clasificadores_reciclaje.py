"""
Comparacion de Clasificadores - Clasificacion de Residuos Reciclables
Entrega 3 - Vision Artificial (Start-Up)
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ============================================================
# 1. CONFIGURACION
# ============================================================

# Debe coincidir con OUTPUT_DIR del script de extraccion
FEATURES_DIR = r"C:\Users\Tp\Downloads\Curso_computervision\features_reciclaje"

FEATURE_FILES = {
    'HOG': 'features_hog.npz',
    'LBP': 'features_lbp.npz',
    'GLCM': 'features_glcm.npz',
    'FUSION (HOG+LBP+GLCM)': 'features_fusion.npz'
}

PROMEDIO_METRICAS = 'weighted'  # 'weighted' (recomendado) o 'macro'

# ============================================================
# 2. DEFINICION DE CLASIFICADORES (igual al original)
# ============================================================

models = {
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
    'SVM (RBF)': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

# ============================================================
# 3. EJECUCION: ENTRENAR Y EVALUAR CADA COMBINACION
# ============================================================

results = []
matrices_confusion = {}  # guarda la matriz de confusion de cada combinacion, para poder graficarla despues
clases = None

print("Iniciando Benchmark de Modelos y Descriptores (Clasificacion de Residuos)...\n")

for feat_name, file_name in FEATURE_FILES.items():
    path = os.path.join(FEATURES_DIR, file_name)
    data = np.load(path, allow_pickle=True)

    X_train, X_val = data['X_train'], data['X_val']
    y_train, y_val = data['y_train'], data['y_val']

    if clases is None and 'clases' in data:
        clases = list(data['clases'])

    for model_name, model in models.items():
        t0 = time.time()

        # Entrenamiento
        model.fit(X_train, y_train)

        # Prediccion
        y_pred = model.predict(X_val)
        t_exec = time.time() - t0

        # Metricas de evaluacion (multiclase: promedio ponderado por soporte de cada clase)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average=PROMEDIO_METRICAS, zero_division=0)
        rec = recall_score(y_val, y_pred, average=PROMEDIO_METRICAS, zero_division=0)
        f1 = f1_score(y_val, y_pred, average=PROMEDIO_METRICAS, zero_division=0)

        results.append({
            'Descriptor': feat_name,
            'Modelo': model_name,
            'Dimensiones': X_train.shape[1],
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Tiempo (s)': t_exec
        })

        matrices_confusion[(feat_name, model_name)] = confusion_matrix(y_val, y_pred)

if clases is None:
    clases = [str(i) for i in sorted(set(y_val.tolist()))]

df_results = pd.DataFrame(results)

print("=" * 95)
print("TABLA CONSOLIDADA DE BENCHMARK (CLASIFICACION DE RESIDUOS RECICLABLES - 6 CLASES)")
print("=" * 95)
print(df_results.to_string(index=False))

# ============================================================
# 4. GRAFICA COMPARATIVA DE F1-SCORE (igual al original)
# ============================================================

plt.figure(figsize=(12, 6))

df_pivot = df_results.pivot(index='Descriptor', columns='Modelo', values='F1-Score')

ax = df_pivot.plot(kind='bar', figsize=(11, 6), colormap='Set2', edgecolor='black')
plt.title("Comparativa de Desempeno (F1-Score Ponderado) segun Extractor y Algoritmo ML", fontsize=13)
plt.ylabel("F1-Score")
plt.xlabel("Extractor de Caracteristicas")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Algoritmos ML", loc='lower right')

for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f"{p.get_height():.2f}",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5),
                     textcoords='offset points', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================
# 5. MATRIZ DE CONFUSION DE LA MEJOR COMBINACION
#    (nuevo respecto al original: con 6 clases, el F1 global no dice
#    QUE confunde el modelo -- esto es clave para el pitch de negocio,
#    ej. "el sistema a veces confunde vidrio con plastico, por eso
#    recomendamos un sensor adicional en esa estacion de la banda").
# ============================================================

mejor_fila = df_results.loc[df_results['F1-Score'].idxmax()]
mejor_combo = (mejor_fila['Descriptor'], mejor_fila['Modelo'])
mejor_matriz = matrices_confusion[mejor_combo]

print(f"\nMejor combinacion: {mejor_combo[0]} + {mejor_combo[1]} "
      f"(F1-Score ponderado = {mejor_fila['F1-Score']:.3f})")

plt.figure(figsize=(7, 6))
plt.imshow(mejor_matriz, cmap='Blues')
plt.title(f"Matriz de Confusion - {mejor_combo[0]} + {mejor_combo[1]}")
plt.colorbar(label="Numero de imagenes")
plt.xticks(range(len(clases)), clases, rotation=45, ha='right')
plt.yticks(range(len(clases)), clases)
plt.xlabel("Clase predicha")
plt.ylabel("Clase real")

for i in range(mejor_matriz.shape[0]):
    for j in range(mejor_matriz.shape[1]):
        color = "white" if mejor_matriz[i, j] > mejor_matriz.max() / 2 else "black"
        plt.text(j, i, str(mejor_matriz[i, j]), ha="center", va="center", color=color)

plt.tight_layout()
plt.show()
