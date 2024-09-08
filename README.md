# Dashboard de Predicción del Iris Dataset

## Descripción del Proyecto

Este proyecto es una aplicación web interactiva creada con **Streamlit** que permite visualizar gráficos y realizar predicciones de especies de flores del **Iris Dataset**. El usuario puede ajustar las características de las flores a través de sliders y ver en tiempo real a qué especie pertenece la flor según un modelo de clasificación basado en **Random Forest**.

### Características:
- **Visualización de datos**: Muestra una tabla con las primeras filas del Iris Dataset.
- **Visualización gráfica**: Permite visualizar gráficos interactivos, incluyendo:
  - Gráfico de pares (**pairplot**) para explorar las relaciones entre las características.
  - Histogramas de distribución para cada característica individual.
- **Predicción**: Predice la especie de una flor (Iris Setosa, Iris Versicolor o Iris Virginica) en función de las características de longitud y ancho de sépalo y pétalo.
- **Modelo**: Utiliza un **RandomForestClassifier** entrenado con los datos del Iris Dataset para realizar las predicciones.

## Requisitos previos

Antes de ejecutar este proyecto, necesitas tener instalados:
- **Python 3.7 o superior**
- **Pip** (el gestor de paquetes de Python)

### Dependencias

Las dependencias necesarias para el proyecto incluyen:
- **streamlit**
- **pandas**
- **seaborn**
- **matplotlib**
- **scikit-learn**

Estas dependencias se pueden instalar ejecutando los pasos descritos a continuación.

## Instalación

1. **Clonar el repositorio** o descargar el código fuente en tu máquina local.
2. **Crear un entorno virtual** (opcional pero recomendado):

   ```bash
   python -m venv env

Luego, activa el entorno virtual:

En Windows:
.\env\Scripts\Activate

En macOS/Linux:
source env/bin/activate

3. Instalar las dependencias:

Ejecuta el siguiente comando para instalar todas las librerías necesarias:

   ```bash
pip install -r requirements.txt
```

Si no tienes el archivo requirements.txt, puedes instalar las dependencias manualmente:

   ```bash
pip install streamlit pandas seaborn matplotlib scikit-learn
```

4. Ejecutar la aplicación:

Después de instalar las dependencias, puedes ejecutar la aplicación Streamlit con:

   ```bash
streamlit run app.py
```

5. Abre el navegador y ve a la URL proporcionada por Streamlit (normalmente http://localhost:8501).

## Explicación del Código

El proyecto está dividido en varias funciones, cada una de las cuales realiza una tarea específica:

### cargar_datos()

Carga el **Iris Dataset** utilizando `load_iris()` de Scikit-learn y lo convierte en un DataFrame de Pandas. También añade una columna para las especies.

### mostrar_datos(df_iris)

Muestra las primeras filas del dataset en la interfaz de usuario usando `st.write()` de Streamlit.

### visualizacion_graficos(df_iris)

Permite al usuario seleccionar entre dos tipos de gráficos interactivos:

- **Gráfico de pares (pairplot)** para observar las relaciones entre las características del dataset.
- **Histograma de distribución** de una característica seleccionada.

### user_input_features(df_iris)

Muestra sliders en la barra lateral para que el usuario introduzca las características de la flor (longitud y ancho de sépalo y pétalo). Estos valores se utilizan para hacer predicciones.

### predecir_especie(model, input_df, iris)

Realiza la predicción de la especie basada en las características proporcionadas por el usuario usando un modelo de **Random Forest**.

### main()

Es la función principal que orquesta el flujo de la aplicación. Carga los datos, entrena el modelo, permite la interacción del usuario y muestra las predicciones y gráficos.

## Explicación del Iris Dataset

El **Iris Dataset** es un conjunto de datos muy conocido en el campo del aprendizaje automático. Este dataset incluye 150 muestras de flores de tres especies diferentes de Iris (**Iris Setosa**, **Iris Versicolor**, **Iris Virginica**). Cada muestra tiene 4 características:

- **Longitud del sépalo** (en cm)
- **Ancho del sépalo** (en cm)
- **Longitud del pétalo** (en cm)
- **Ancho del pétalo** (en cm)

### Estructura del dataset:

- **150 muestras** (50 muestras por cada una de las 3 especies).
- **4 características numéricas** que describen las flores.
- **1 etiqueta de clase** que representa la especie de la flor.

El objetivo es, a partir de las características de la flor, predecir a qué especie pertenece. Este es un problema clásico de clasificación multiclase.

### Ejemplo de los datos:

| Longitud del sépalo (cm) | Ancho del sépalo (cm) | Longitud del pétalo (cm) | Ancho del pétalo (cm) | Especie         |
|--------------------------|-----------------------|--------------------------|-----------------------|-----------------|
| 5.1                      | 3.5                   | 1.4                      | 0.2                   | Iris Setosa      |
| 7.0                      | 3.2                   | 4.7                      | 1.4                   | Iris Versicolor  |
| 6.3                      | 3.3                   | 6.0                      | 2.5                   | Iris Virginica   |


### Uso del Dataset en el Proyecto

En este proyecto, el **Iris Dataset** se utiliza para entrenar un modelo de clasificación **Random Forest**, que predice la especie de una flor en función de las características proporcionadas por el usuario. 

El modelo se entrena utilizando la función `train_test_split()` de **Scikit-learn** para dividir el dataset en un conjunto de entrenamiento y un conjunto de prueba. 

Luego, las predicciones se generan para los datos proporcionados por el usuario a través de la interfaz de **Streamlit**.

