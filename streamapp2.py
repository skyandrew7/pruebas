import streamlit as st
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Título de la app
st.title("Predicción de Series de Tiempo con Prophet")

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube un archivo CSV con datos históricos", type=["csv"])

if uploaded_file:
    # Cargar los datos
    df = pd.read_csv(uploaded_file)
    st.write("Datos cargados:")
    st.dataframe(df.head())

    # Seleccionar industria
    industry_options = df['industry'].unique()
    industry_type = st.selectbox("Selecciona la industria:", options=industry_options)

    # Seleccionar columna para predecir
    column_options = ['trips_per_day', 'total_co2_emission', 'vehicles_per_day', 
                      'total_trips', 'total_amount', 'farebox_per_day_per_distance', 
                      'farebox_per_day', 'unique_drivers', 'unique_vehicles', 'avg_trip_distance']
    column_name = st.selectbox("Selecciona la columna objetivo:", options=column_options)

    # Seleccionar parámetros de predicción
    periodos = st.slider("Selecciona el horizonte de predicción (en períodos):", min_value=1, max_value=365, value=30)
    frecuencia = st.selectbox("Selecciona la frecuencia de predicción:", options=["D", "W", "M", "YE"], index=0)

    # Preparar los datos para Prophet
    def cargar_y_preparar_datos(df, industry_type, column_name):
        df_filtered = df[df['industry'] == industry_type].copy()
        df_filtered = df_filtered[df_filtered[column_name] >= 0]
        df_prophet = df_filtered[['date', column_name]].rename(columns={'date': 'ds', column_name: 'y'})
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
        return df_prophet

    df_prophet = cargar_y_preparar_datos(df, industry_type, column_name)

    if not df_prophet.empty:
        # Mostrar datos filtrados
        st.write("Datos filtrados para Prophet:")
        st.dataframe(df_prophet.head())

        # Graficar datos originales
        st.subheader("Datos históricos")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], mode='lines+markers', name='Datos'))
        fig.update_layout(title="Datos históricos", xaxis_title="Fecha", yaxis_title=column_name)
        st.plotly_chart(fig)

        # Entrenar y predecir con Prophet
        st.subheader("Resultados de predicción")
        model = Prophet(changepoint_prior_scale=0.04, seasonality_prior_scale=10.0, seasonality_mode='additive')
        model.add_seasonality(name='monthly', period=12, fourier_order=2)
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=periodos, freq=frecuencia)
        forecast = model.predict(future)

        # Se grafica la predicción
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], mode='lines+markers', name='Datos históricos'))
        fig_pred.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode='lines', name='Predicción'))
        fig_pred.add_trace(go.Scatter(x=forecast['ds'], 
                                      y=forecast['yhat_upper'], 
                                      mode='lines', 
                                      line=dict(dash='dot', color='gray'), 
                                      name='Límite superior'))
        fig_pred.add_trace(go.Scatter(x=forecast['ds'], 
                                      y=forecast['yhat_lower'], 
                                      mode='lines', 
                                      line=dict(dash='dot', color='gray'), 
                                      name='Límite inferior'))
        fig_pred.update_layout(title=f"Predicción para {column_name} ({industry_type})", 
                               xaxis_title="Fecha", yaxis_title=column_name)
        st.plotly_chart(fig_pred)

        # Mostramos los componentes de la predicción
        st.subheader("Componentes de la predicción")
        fig_components = model.plot_components(forecast)
        st.pyplot(fig_components)

        # Descargamos las predicciones como CSV
        st.subheader("Descargar predicciones")
        csv = forecast.to_csv(index=False)
        st.download_button(label="Descargar predicciones", data=csv, file_name="predicciones.csv", mime="text/csv")
    else:
        st.error("No se encontraron datos para esta combinación de industria y columna.")
else:
    st.info("Sube un archivo CSV para empezar.")
