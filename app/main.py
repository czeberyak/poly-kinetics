import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# --- НАСТРОЙКА ПУТЕЙ И ИМПОРТОВ (Решает проблему ModuleNotFoundError) ---
# Находим абсолютный путь к корню проекта (poly-kinetics)
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[1]  # Выходим на уровень poly-kinetics/

# Добавляем именно корень проекта в sys.path (чтобы src импортировался как пакет)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Теперь импорты структуры модели пройдут успешно во время десериализации joblib
from src.arrhenius_model import ArrheniusKinetics, PhysicsInformedTransformer

# Определение путей к файлам моделей на основе корня проекта
MODEL_JSON_PATH = project_root / 'models' / 'arrhenius_params.json'
MODEL_JOBLIB_PATH = project_root / 'models' / 'random_forest_model.joblib'

# Настройка страницы
st.set_page_config(page_title="PU Kinetics Predictor", layout="centered", page_icon="🧪")
st.title("🧪 Предиктор времени жизни ПУ системы")
st.markdown("Сравнение физической модели Аррениуса и ML-прогноза (Random Forest)")

# --- ЗАГРУЗКА МОДЕЛЕЙ ---
@st.cache_resource
def load_models():
    """Загружает физическую и ML модели один раз при старте."""
    # 1. Физическая модель
    physics_model = ArrheniusKinetics()
    if MODEL_JSON_PATH.exists():
        physics_model.load_model(path=MODEL_JSON_PATH)
    else:
        st.warning(f"⚠️ Физическая модель не найдена по пути: {MODEL_JSON_PATH}")
        physics_model = None
    
    # 2. ML-пайплайн (загружаем сохраненный на Шаге 2 rf_pipeline.joblib)
    ml_pipeline = None
    if MODEL_JOBLIB_PATH.exists():
        try:
            ml_pipeline = joblib.load(MODEL_JOBLIB_PATH)
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке ML-пайплайна: {e}")
    else:
        st.warning(f"⚠️ ML-пайплайн не найден по пути: {MODEL_JOBLIB_PATH}")
        
    return physics_model, ml_pipeline

physics_model, ml_pipeline = load_models()

# --- ИНТЕРФЕЙС ВВОДА ---
st.sidebar.header("Параметры рецептуры")

temp_c = st.sidebar.slider("Температура (°C)", 15.0, 35.0, 25.0, 0.5)
humidity = st.sidebar.slider("Влажность (%)", 30.0, 80.0, 50.0, 1.0)
catalyst = st.sidebar.slider("Катализатор (%)", 0.05, 0.30, 0.15, 0.01)
nco_oh = st.sidebar.slider("NCO/OH Ratio", 1.00, 1.15, 1.05, 0.01)

# Кнопка прогноза
if st.button("Рассчитать прогноз", type="primary"):
    if physics_model is None or ml_pipeline is None:
        st.error("Не все модели загружены. Проверьте предупреждения выше.")
    else:
        # 1. Прогноз от физической модели (принимает только температуру)
        pred_physics = physics_model.predict(temp_c)
        
        # 2. Прогноз от ML-пайплайна (передаем только сырые 4 признака)
        input_data = pd.DataFrame({
            'Temperature_C': [temp_c],
            'Humidity_pct': [humidity],
            'Catalyst_pct': [catalyst],
            'NCO_OH_Ratio': [nco_oh]
        })
        
        # Пайплайн сам извлечет термодинамический признак и отмасштабирует данные
        pred_ml = ml_pipeline.predict(input_data)[0]
        
        # --- ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Физическая модель (Аррениус)", 
                value=f"{pred_physics:.1f} мин",
                delta_color="off"
            )
            st.caption("Основана на найденных параметрах Ea/R")
            
        with col2:
            st.metric(
                label="ML Модель (Random Forest)", 
                value=f"{pred_ml:.1f} мин",
                delta=f"{pred_ml - pred_physics:+.1f} мин от физики",
                delta_color="normal"
            )
            st.caption("Обучена с помощью Physics-Informed Pipeline")
            
        # Сравнительный график
        st.divider()
        st.subheader("Сравнение прогнозов")
        chart_data = pd.DataFrame({
            'Метод': ['Физика', 'ML'],
            'Время жизни (мин)': [pred_physics, pred_ml]
        })
        st.bar_chart(chart_data.set_index('Метод'), height=250, use_container_width=True)
        
        st.info(f"""
        **Анализ:** При температуре {temp_c}°C и катализаторе {catalyst}% 
        расхождение между физическим расчетом и ML-моделью составляет {abs(pred_ml - pred_physics):.1f} минут.
        """)