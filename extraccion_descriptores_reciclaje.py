"""
Extraccion de Descriptores (HOG, LBP, GLCM) - Clasificacion de Residuos Reciclables
Entrega 3 - Vision Artificial (Start-Up)
"""
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importaciones de skimage para descriptores de caracteristicas
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

# ============================================================
# 1. CONFIGURACION
# ============================================================

# Ajusta esta ruta a donde descomprimiste el dataset de Kaggle.
RUTA_DATASET = r"C:\Users\Usuario\Downloads\garbage-classification"

CLASES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

TAMANO_IMAGEN = (128, 128)
EXTENSIONES = ('.jpg', '.jpeg', '.png', '.bmp')

# Limite de imagenes por clase, para que la extraccion (sobre todo GLCM,
# que es la mas costosa) no tarde demasiado. El dataset real queda
# balanceado usando el minimo entre este limite y la clase con menos
# imagenes disponibles (normalmente "trash").
MAX_IMAGENES_POR_CLASE = 300

# Dos clases usadas solo para la visualizacion detallada de descriptores
# (puedes cambiarlas por cualquier par de CLASES que quieras comparar).
CLASE_EJEMPLO_A = 'cardboard'
CLASE_EJEMPLO_B = 'plastic'

# Carpeta donde se guardaran los .npz con las caracteristicas extraidas
OUTPUT_DIR = r"C:\Users\Usuario\Downloads\garbage-classification"


# ============================================================
# 2. CARGA DE IMAGENES POR CLASE
# ============================================================

def encontrar_carpeta_clase(raiz, nombre_clase):
    """
    Busca recursivamente, dentro de 'raiz', la carpeta cuyo nombre
    coincide (sin importar mayusculas/minusculas) con 'nombre_clase'.
    Evita tener que adivinar cuantos niveles de carpetas anidadas trae
    el .zip descargado de Kaggle.
    """
    for carpeta_actual, subcarpetas, _ in os.walk(raiz):
        for sub in subcarpetas:
            if sub.lower() == nombre_clase.lower():
                return os.path.join(carpeta_actual, sub)
    return None


def cargar_imagenes_validas(path, n):
    archivos = sorted(f for f in os.listdir(path) if f.lower().endswith(EXTENSIONES))
    imgs = []
    for f in archivos:
        img = cv2.imread(os.path.join(path, f))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
        if len(imgs) == n:
            break
    return imgs


def normalizar_imagenes(imgs, size=TAMANO_IMAGEN):
    imgs_norm = []
    for img in imgs:
        if img is not None:
            img_resized = cv2.resize(img, size)
            img_norm = img_resized.astype("float32") / 255.0
            imgs_norm.append(img_norm)
    return np.array(imgs_norm)


# --- Ubicar la carpeta real de cada clase y contar imagenes disponibles ---
rutas_clase = {}
conteos = {}
for clase in CLASES:
    ruta = encontrar_carpeta_clase(RUTA_DATASET, clase)
    if ruta is None:
        raise FileNotFoundError(
            f"No se encontro la carpeta de la clase '{clase}' dentro de "
            f"'{RUTA_DATASET}'. Revisa RUTA_DATASET."
        )
    rutas_clase[clase] = ruta
    conteos[clase] = len([f for f in os.listdir(ruta) if f.lower().endswith(EXTENSIONES)])

print("Imagenes disponibles por clase:")
for clase in CLASES:
    print(f"  {clase:10s}: {conteos[clase]}")

# Dataset balanceado: mismo numero de imagenes en cada clase (igual que el
# original con 1011 gatos / 1011 perros), usando el minimo disponible
# acotado por MAX_IMAGENES_POR_CLASE.
n_por_clase = min(min(conteos.values()), MAX_IMAGENES_POR_CLASE)
print(f"\nUsando {n_por_clase} imagenes por clase (dataset balanceado).\n")

imagenes_por_clase = []
etiquetas = []
for idx_clase, clase in enumerate(CLASES):
    imgs_clase = cargar_imagenes_validas(rutas_clase[clase], n_por_clase)
    imgs_norm = normalizar_imagenes(imgs_clase)
    imagenes_por_clase.append(imgs_norm)
    etiquetas.extend([idx_clase] * len(imgs_norm))
    print(f"  Cargadas {len(imgs_norm)} imagenes de '{clase}'")

X = np.concatenate(imagenes_por_clase, axis=0)
y = np.array(etiquetas)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nDataset listo -> Train: {X_train.shape}, Validacion: {X_val.shape}")


# ============================================================
# 3. EXTRACCION DE DESCRIPTORES (identico al original: funciones
#    por-imagen, no dependen de cuantas clases haya)
# ============================================================

# HOG (Histogram of Oriented Gradients)
def extraer_hog(imagenes):
    hog_features = []
    for img in imagenes:
        img_gray = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
        features = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
        hog_features.append(features)
    return np.array(hog_features)


# LBP (Local Binary Patterns)
def extraer_lbp(imagenes, P=8, R=1):
    lbp_features = []
    for img in imagenes:
        img_gray = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(img_gray, P=P, R=R, method="uniform")
        # Construccion del histograma de patrones invariantes
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        hist = hist.astype("float32")
        hist /= (hist.sum() + 1e-7)  # Normalizacion del histograma LBP
        lbp_features.append(hist)
    return np.array(lbp_features)


# GLCM (Gray-Level Co-occurrence Matrix)
def extraer_glcm(imagenes, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]):
    glcm_features = []
    propiedades = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for img in imagenes:
        img_gray = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
        # Matriz de coocurrencia (escalada a 64 niveles de gris para acelerar el calculo)
        img_gray_q = (img_gray // 4).astype(np.uint8)
        glcm = graycomatrix(img_gray_q, distances=distances, angles=angles, levels=64,
                             symmetric=True, normed=True)

        vector_img = []
        for prop in propiedades:
            valores = graycoprops(glcm, prop).ravel()
            vector_img.extend(valores)
        glcm_features.append(vector_img)
    return np.array(glcm_features)


print("\nExtrayendo HOG...")
X_train_hog = extraer_hog(X_train)
X_val_hog = extraer_hog(X_val)

print("Extrayendo LBP...")
X_train_lbp = extraer_lbp(X_train)
X_val_lbp = extraer_lbp(X_val)

print("Extrayendo GLCM...")
X_train_glcm = extraer_glcm(X_train)
X_val_glcm = extraer_glcm(X_val)

print(f"\nDimensiones resultantes:")
print(f"HOG  -> Train: {X_train_hog.shape}")
print(f"LBP  -> Train: {X_train_lbp.shape}")
print(f"GLCM -> Train: {X_train_glcm.shape}")


# ============================================================
# 4. VISUALIZACION
# ============================================================

# --- 4a. Un ejemplo de cada una de las 6 clases (bueno para el video-pitch) ---
fig0, axes0 = plt.subplots(1, len(CLASES), figsize=(3 * len(CLASES), 3))
for i, clase in enumerate(CLASES):
    idx_ejemplo = np.where(y_val == i)[0]
    if len(idx_ejemplo) > 0:
        axes0[i].imshow(X_val[idx_ejemplo[0]])
    axes0[i].set_title(clase)
    axes0[i].axis('off')
plt.suptitle("Una muestra de cada clase del dataset de reciclaje")
plt.tight_layout()
plt.show()

# --- 4b. Comparacion detallada de descriptores entre dos clases representativas ---
idx_a = np.where(y_val == CLASES.index(CLASE_EJEMPLO_A))[0][0]
idx_b = np.where(y_val == CLASES.index(CLASE_EJEMPLO_B))[0][0]

img_a_gray = cv2.cvtColor((X_val[idx_a] * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
img_b_gray = cv2.cvtColor((X_val[idx_b] * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)

# Calculo de mapas visuales
_, hog_a_vis = hog(img_a_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
_, hog_b_vis = hog(img_b_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

lbp_a_vis = local_binary_pattern(img_a_gray, P=8, R=1, method="uniform")
lbp_b_vis = local_binary_pattern(img_b_gray, P=8, R=1, method="uniform")

# GLCM (Matriz 2D a distancia 1, angulo 0 grados)
a_q = (img_a_gray // 4).astype(np.uint8)
b_q = (img_b_gray // 4).astype(np.uint8)

glcm_a_vis = graycomatrix(a_q, distances=[1], angles=[0], levels=64, symmetric=True, normed=True)[:, :, 0, 0]
glcm_b_vis = graycomatrix(b_q, distances=[1], angles=[0], levels=64, symmetric=True, normed=True)[:, :, 0, 0]

fig1, axes1 = plt.subplots(4, 2, figsize=(11, 13))

axes1[0, 0].imshow(X_val[idx_a])
axes1[0, 0].set_title(f"{CLASE_EJEMPLO_A.capitalize()} (Original)")
axes1[0, 0].axis('off')

axes1[0, 1].imshow(X_val[idx_b])
axes1[0, 1].set_title(f"{CLASE_EJEMPLO_B.capitalize()} (Original)")
axes1[0, 1].axis('off')

axes1[1, 0].imshow(hog_a_vis, cmap='gray')
axes1[1, 0].set_title(f"HOG - Mapa de Gradientes ({CLASE_EJEMPLO_A})")
axes1[1, 0].axis('off')

axes1[1, 1].imshow(hog_b_vis, cmap='gray')
axes1[1, 1].set_title(f"HOG - Mapa de Gradientes ({CLASE_EJEMPLO_B})")
axes1[1, 1].axis('off')

axes1[2, 0].imshow(lbp_a_vis, cmap='gray')
axes1[2, 0].set_title(f"LBP - Mapa de Micro-textura ({CLASE_EJEMPLO_A})")
axes1[2, 0].axis('off')

axes1[2, 1].imshow(lbp_b_vis, cmap='gray')
axes1[2, 1].set_title(f"LBP - Mapa de Micro-textura ({CLASE_EJEMPLO_B})")
axes1[2, 1].axis('off')

axes1[3, 0].imshow(glcm_a_vis, cmap='viridis')
axes1[3, 0].set_title(f"GLCM - Matriz de Coocurrencia 0 grados ({CLASE_EJEMPLO_A})")
axes1[3, 0].set_xlabel("Nivel de gris j")
axes1[3, 0].set_ylabel("Nivel de gris i")

axes1[3, 1].imshow(glcm_b_vis, cmap='viridis')
axes1[3, 1].set_title(f"GLCM - Matriz de Coocurrencia 0 grados ({CLASE_EJEMPLO_B})")
axes1[3, 1].set_xlabel("Nivel de gris j")
axes1[3, 1].set_ylabel("Nivel de gris i")

plt.figure(fig1.number)
plt.tight_layout()
plt.show()


# ============================================================
# 5. FUSION Y ESCALADO OBLIGATORIO (Z-score)
# ============================================================

X_train_fusion = np.hstack((X_train_hog, X_train_lbp, X_train_glcm))
X_val_fusion = np.hstack((X_val_hog, X_val_lbp, X_val_glcm))

scaler = StandardScaler()
X_train_fusion_scaled = scaler.fit_transform(X_train_fusion)
X_val_fusion_scaled = scaler.transform(X_val_fusion)

fusion_a_vis = X_val_fusion_scaled[idx_a].reshape(1, -1)
fusion_b_vis = X_val_fusion_scaled[idx_b].reshape(1, -1)

fig2, axes2 = plt.subplots(2, 2, figsize=(11, 6))

axes2[0, 0].imshow(X_val[idx_a])
axes2[0, 0].set_title(f"{CLASE_EJEMPLO_A.capitalize()} (Entrada)")
axes2[0, 0].axis('off')

axes2[0, 1].imshow(X_val[idx_b])
axes2[0, 1].set_title(f"{CLASE_EJEMPLO_B.capitalize()} (Entrada)")
axes2[0, 1].axis('off')

axes2[1, 0].imshow(fusion_a_vis, aspect='auto', cmap='coolwarm', vmin=-2, vmax=2)
axes2[1, 0].set_title(f"Firma Fusionada ({fusion_a_vis.shape[1]} dims) - {CLASE_EJEMPLO_A}")
axes2[1, 0].set_yticks([])
axes2[1, 0].set_xlabel("Indices de caracteristicas (Normalizadas Z-Score)")

axes2[1, 1].imshow(fusion_b_vis, aspect='auto', cmap='coolwarm', vmin=-2, vmax=2)
axes2[1, 1].set_title(f"Firma Fusionada ({fusion_b_vis.shape[1]} dims) - {CLASE_EJEMPLO_B}")
axes2[1, 1].set_yticks([])
axes2[1, 1].set_xlabel("Indices de caracteristicas (Normalizadas Z-Score)")

plt.figure(fig2.number)
plt.tight_layout()
plt.show()

print(f"Vector Fusionado y Escalado -> Train: {X_train_fusion_scaled.shape}, Val: {X_val_fusion_scaled.shape}")


# ============================================================
# 6. GUARDADO DE CARACTERISTICAS (INDIVIDUALES Y FUSIONADAS)
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Escaladores independientes para cada conjunto individual
scaler_hog = StandardScaler()
X_tr_hog_sc = scaler_hog.fit_transform(X_train_hog)
X_va_hog_sc = scaler_hog.transform(X_val_hog)

scaler_lbp = StandardScaler()
X_tr_lbp_sc = scaler_lbp.fit_transform(X_train_lbp)
X_va_lbp_sc = scaler_lbp.transform(X_val_lbp)

scaler_glcm = StandardScaler()
X_tr_glcm_sc = scaler_glcm.fit_transform(X_train_glcm)
X_va_glcm_sc = scaler_glcm.transform(X_val_glcm)

# Se guarda tambien 'clases' en cada archivo: el script de clasificacion
# lo usa para mostrar nombres reales (no solo 0-5) en reportes y matriz
# de confusion.
clases_arr = np.array(CLASES)

np.savez_compressed(os.path.join(OUTPUT_DIR, "features_hog.npz"),
                     X_train=X_tr_hog_sc, X_val=X_va_hog_sc,
                     y_train=y_train, y_val=y_val, clases=clases_arr)

np.savez_compressed(os.path.join(OUTPUT_DIR, "features_lbp.npz"),
                     X_train=X_tr_lbp_sc, X_val=X_va_lbp_sc,
                     y_train=y_train, y_val=y_val, clases=clases_arr)

np.savez_compressed(os.path.join(OUTPUT_DIR, "features_glcm.npz"),
                     X_train=X_tr_glcm_sc, X_val=X_va_glcm_sc,
                     y_train=y_train, y_val=y_val, clases=clases_arr)

np.savez_compressed(os.path.join(OUTPUT_DIR, "features_fusion.npz"),
                     X_train=X_train_fusion_scaled, X_val=X_val_fusion_scaled,
                     y_train=y_train, y_val=y_val, clases=clases_arr)

print("\nCaracteristicas guardadas exitosamente en:", OUTPUT_DIR)
