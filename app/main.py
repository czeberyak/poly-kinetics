import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Добавляем корневую папку проекта в путь, чтобы видеть src/
sys.path.append(os.path.abspath('../src'))
from arrhenius_model import ArrheniusKinetics

# Настройка страницы
st.set_page_config(page_title="PU Kinetics Predictor", layout="centered")
st.title("🧪 Предиктор времени жизни ПУ системы")
st.markdown("Сравнение физической модели Аррениуса и ML-прогноза (Random Forest)")

# --- ЗАГРУЗКА МОДЕЛЕЙ ---
@st.cache_resource
def load_models():
    """Загружает физическую и ML модели один раз при старте."""
    # 1. Физическая модель
    physics_model = ArrheniusKinetics()
    try:
        physics_model.load_model(path='../models/arrhenius_params.json')
    except FileNotFoundError:
        st.warning("⚠️ Физическая модель не найдена. Обучите её сначала в ноутбуке!")
        physics_model = None
    
    # 2. ML Модель
    ml_model = None
    try:
        ml_model = joblib.load('../models/random_forest_model.joblib')
    except FileNotFoundError:
        st.warning("⚠️ ML модель не найдена. Сохраните её через joblib!")
        
    return physics_model, ml_model

physics_model, ml_model = load_models()

# --- ИНТЕРФЕЙС ВВОДА ---
st.sidebar.header("Параметры рецептуры")

temp_c = st.sidebar.slider("Температура (°C)", 15.0, 35.0, 25.0, 0.5)
humidity = st.sidebar.slider("Влажность (%)", 30.0, 80.0, 50.0, 1.0)
catalyst = st.sidebar.slider("Катализатор (%)", 0.05, 0.30, 0.15, 0.01)
nco_oh = st.sidebar.slider("NCO/OH Ratio", 1.00, 1.15, 1.05, 0.01)

# Кнопка прогноза
if st.button("Рассчитать прогноз"):
    if physics_model is None or ml_model is None:
        st.error("Не все модели загружены. Проверьте предупреждения выше.")
    else:
        # 1. Прогноз от физической модели
        pred_physics = physics_model.predict(temp_c)
        
        # 2. Прогноз от ML модели
        input_data = pd.DataFrame({
            'Temperature_C': [temp_c],
            'Humidity_pct': [humidity],
            'Catalyst_pct': [catalyst],
            'NCO_OH_Ratio': [nco_oh]
        })
        pred_ml = ml_model.predict(input_data)[0]
        
        # --- ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Физическая модель (Аррениус)", 
                value=f"{pred_physics:.1f} мин",
                delta_color="off"
            )
            st.caption("Основана на уравнении кинетики и найденных Ea/R")
            
        with col2:
            st.metric(
                label="ML Модель (Random Forest)", 
                value=f"{pred_ml:.1f} мин",
                delta=f"{pred_ml - pred_physics:+.1f}",
                delta_color="normal"
            )
            st.caption("Обучена на синтетических данных с учетом всех фичей")
            
        # Сравнительный график
        st.divider()
        st.subheader("Сравнение прогнозов")
        chart_data = pd.DataFrame({
            'Метод': ['Физика', 'ML'],
            'Время жизни (мин)': [pred_physics, pred_ml]
        })
        st.bar_chart(chart_data.set_index('Метод'), height=300)
        
        st.info(f"""
        **Анализ:** При температуре {temp_c}°C и катализаторе {catalyst}% 
        расхождение между моделями составляет {abs(pred_ml - pred_physics):.1f} минут.
        """)