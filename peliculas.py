import pandas as pd
from sklearn.preprocessing import StandardScaler

import pandas as pd

# 1. EXTRACCIÓN
df_raw = pd.read_csv("src/HollywoodMovies.csv")

# 2. TRANSFORMACIÓN
# a. Seleccionar el identificador y las variables numéricas clave
columnas_numericas = [
    "Budget",
    "WorldGross",
    "RottenTomatoes",
    "AudienceScore",
    "Profitability",
]
df_clean = df_raw[["Movie"] + columnas_numericas].copy()

# b. Convertir columnas a formato numérico (por si hay caracteres extraños)
for col in columnas_numericas:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# c. Eliminar filas que tengan valores nulos en estas características numéricas
df_clean = df_clean.dropna(subset=columnas_numericas).reset_index(drop=True)

# d. Normalizar / Escalar los datos (media = 0, desviación estándar = 1)
scaler = StandardScaler()
datos_escalados = scaler.fit_transform(df_clean[columnas_numericas])

# Crear un DataFrame con las características escaladas
df_scaled = pd.DataFrame(datos_escalados, columns=columnas_numericas)
df_scaled["Movie"] = df_clean["Movie"]

# 3. CARGA 
df_clean.to_csv("src/HollywoodMovies_Clean.csv", index=False)
df_scaled.to_csv("src/HollywoodMovies_Scaled.csv", index=False)

print("¡Proceso de ETL completado con éxito!")
print(f"Total de películas procesadas sin nulos: {len(df_clean)}")
print("\nMuestra de datos normalizados:")
print(df_scaled.head())

# Agregar al final de peliculas.py
print(f"Filas originales: {len(df_raw)}")
print(f"Filas limpias tras dropna: {len(df_clean)}")
