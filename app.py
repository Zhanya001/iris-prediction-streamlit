import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def cargar_datos():
    """
    Carga el Iris Dataset y lo transforma en un DataFrame de Pandas.
    
    Returns:
        DataFrame: DataFrame del Iris Dataset con una columna adicional para la especie.
    """
    iris = load_iris()
    df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df_iris['target'] = iris.target
    df_iris['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df_iris, iris

def mostrar_datos(df_iris):
    """
    Muestra una tabla de los datos cargados en la aplicación.

    Args:
        df_iris (DataFrame): DataFrame del Iris Dataset.
    """
    st.header('Datos del Iris Dataset')
    st.write(df_iris.head())

def visualizacion_graficos(df_iris):
    """
    Muestra gráficos interactivos basados en la selección del usuario.

    Args:
        df_iris (DataFrame): DataFrame del Iris Dataset.
    """
    st.header('Visualización de Gráficos')

    # Selección del gráfico
    grafico = st.selectbox("Selecciona el gráfico", ["Pares (pairplot)", "Distribución (Histogram)"])

    # Mostrar gráfico seleccionado
    if grafico == "Pares (pairplot)":
        st.subheader('Pairplot de las características')
        try:
            fig = sns.pairplot(df_iris, hue='species')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error al generar el gráfico: {e}")
    else:
        st.subheader('Distribución de las características')
        feature = st.selectbox("Selecciona la característica para ver la distribución", df_iris.columns[:-2])
        try:
            fig, ax = plt.subplots()
            sns.histplot(df_iris[feature], kde=True, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error al generar la distribución: {e}")

def user_input_features(df_iris):
    """
    Genera los sliders en la barra lateral para ingresar características de entrada y devuelve un DataFrame con estos valores.

    Args:
        df_iris (DataFrame): DataFrame del Iris Dataset.

    Returns:
        DataFrame: DataFrame con las características ingresadas por el usuario.
    """
    st.sidebar.header('Parámetros de entrada')
    try:
        sepal_length = st.sidebar.slider('Longitud del sépalo', float(df_iris['sepal length (cm)'].min()), float(df_iris['sepal length (cm)'].max()))
        sepal_width = st.sidebar.slider('Ancho del sépalo', float(df_iris['sepal width (cm)'].min()), float(df_iris['sepal width (cm)'].max()))
        petal_length = st.sidebar.slider('Longitud del pétalo', float(df_iris['petal length (cm)'].min()), float(df_iris['petal length (cm)'].max()))
        petal_width = st.sidebar.slider('Ancho del pétalo', float(df_iris['petal width (cm)'].min()), float(df_iris['petal width (cm)'].max()))
        
        data = {
            'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width
        }
        features = pd.DataFrame(data, index=[0])
        return features

    except Exception as e:
        st.error(f"Error al ingresar los valores: {e}")
        return None

def predecir_especie(model, input_df, iris):
    """
    Realiza la predicción de la especie basándose en las características ingresadas por el usuario.

    Args:
        model (RandomForestClassifier): Modelo de RandomForest entrenado.
        input_df (DataFrame): Características ingresadas por el usuario.
        iris: Dataset de Iris original para mapear predicciones a nombres de especies.

    Returns:
        Tuple: Predicción de la especie y las probabilidades.
    """
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    return iris.target_names[prediction][0], prediction_proba

def main():
    """
    Función principal para ejecutar la aplicación de Streamlit.
    """
    # Título de la aplicación
    st.title('Dashboard de Predicción del Iris Dataset')

    # Cargar datos
    df_iris, iris = cargar_datos()

    # Mostrar datos del dataset
    mostrar_datos(df_iris)

    # Visualización de gráficos
    visualizacion_graficos(df_iris)

    # Ingreso de características por el usuario
    input_df = user_input_features(df_iris)

    if input_df is not None:
        # Dividir datos para el modelo
        X = df_iris.drop(['target', 'species'], axis=1)
        y = df_iris['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar modelo de Random Forest
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Realizar predicción
        prediccion, probabilidad = predecir_especie(model, input_df, iris)

        # Mostrar predicción
        st.subheader('Parámetros de entrada')
        st.write(input_df)

        st.subheader('Predicción de la especie')
        st.write(prediccion)

        st.subheader('Probabilidad de la predicción')
        st.write(probabilidad)

        # Mostrar precisión del modelo
        st.subheader('Precisión del modelo')
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'Precisión del modelo: {accuracy * 100:.2f}%')

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
