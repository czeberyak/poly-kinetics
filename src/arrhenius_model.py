import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import json
import os
import warnings

class ArrheniusKinetics:
    """
    Класс для моделирования кинетики химических реакций на основе уравнения Аррениуса.
    Поддерживает генерацию данных, обучение, сохранение и загрузку параметров.
    """

    def __init__(self):
        self.Ea_R = None      # Энергия активации / Газовая постоянная
        self.ln_A = None      # Натуральный логарифм предэкспоненциального множителя
        self.is_fitted = False
        self.model_path = "models/arrhenius_params.json"

    @staticmethod
    def _arrhenius_equation(T_K, ln_A, Ea_R):
        """Базовое уравнение Аррениуса: k = A * exp(-Ea / RT)"""
        return np.exp(ln_A - Ea_R / T_K)

    def generate_synthetic_data(self, n_samples=2000, seed=42):
        """Генерирует физически достоверный датасет."""
        np.random.seed(seed)
        
        temp_c = np.random.uniform(15, 35, n_samples)
        temp_k = temp_c + 273.15
        catalyst_pct = np.random.uniform(0.05, 0.3, n_samples)
        humidity = np.random.uniform(30, 80, n_samples)
        nco_oh_ratio = np.random.uniform(1.0, 1.15, n_samples)

        # Физическое ядро (исправленные константы)
        Ea_R_true = 3200
        C_true = 0.0006
        
        base_time = C_true * np.exp(Ea_R_true / temp_k)
        pot_life_pure = (base_time / (catalyst_pct ** 0.6)) - (humidity * 0.2)
        
        noise = np.random.normal(loc=0, scale=8.0, size=n_samples)
        pot_life_real = np.clip(pot_life_pure + noise, 10, 400)

        self.data = pd.DataFrame({
            'Temperature_C': np.round(temp_c, 1),
            'Humidity_pct': np.round(humidity, 1),
            'Catalyst_pct': np.round(catalyst_pct, 3),
            'NCO_OH_Ratio': np.round(nco_oh_ratio, 2),
            'Pot_Life_min': np.round(pot_life_real, 0)
        })
        
        return self.data

    def fit(self, df=None):
        """Обучает модель, находя параметры Ea/R и ln(A)."""
        if df is None and not hasattr(self, 'data'):
            raise ValueError("Нет данных для обучения. Сначала вызовите generate_synthetic_data()")
            
        data = df if df is not None else self.data
        T_K = data['Temperature_C'].values + 273.15
        y = data['Pot_Life_min'].values
        
        mask = y > 0
        T_K_clean = T_K[mask]
        y_clean = y[mask]

        try:
            popt, pcov = curve_fit(self._arrhenius_equation, T_K_clean, y_clean, p0=[10, 3000])
            self.ln_A, self.Ea_R = popt
            self.is_fitted = True
            print(f"Модель обучена успешно!\nНайдено Ea/R: {self.Ea_R:.2f}\nНайдено ln(A): {self.ln_A:.2f}")
        except Exception as e:
            print(f"Ошибка при фиттинге модели: {e}")

    def predict(self, temperature_c):
        """Предсказывает время жизни при заданной температуре."""
        if not self.is_fitted:
            raise RuntimeError("Сначала необходимо обучить модель методом .fit() или загрузить параметры .load_model()")
            
        t_k = temperature_c + 273.15
        return self._arrhenius_equation(t_k, self.ln_A, self.Ea_R)

    def save_model(self, path=None):
        """Сохраняет параметры модели в JSON файл."""
        if not self.is_fitted:
            raise RuntimeError("Нельзя сохранить необученную модель.")
        
        save_path = path or self.model_path
        # Создаем папку models, если её нет
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model_data = {
            "Ea_R": float(self.Ea_R),
            "ln_A": float(self.ln_A),
            "is_fitted": True
        }
        
        with open(save_path, 'w') as f:
            json.dump(model_data, f, indent=4)
        print(f"Параметры модели сохранены в {save_path}")

    def load_model(self, path=None):
        """Загружает параметры модели из JSON файла."""
        load_path = path or self.model_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Файл модели не найден: {load_path}")
            
        with open(load_path, 'r') as f:
            model_data = json.load(f)
            
        self.Ea_R = model_data["Ea_R"]
        self.ln_A = model_data["ln_A"]
        self.is_fitted = model_data["is_fitted"]
        print(f"Модель успешно загружена из {load_path}")