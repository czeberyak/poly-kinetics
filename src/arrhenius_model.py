import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import json
import warnings
from pathlib import Path
from typing import Optional, Union, List

class ArrheniusKinetics:
    """
    Класс для моделирования кинетики химических реакций на основе уравнения Аррениуса.
    Поддерживает генерацию данных, обучение, сохранение и загрузку параметров.
    """

    def __init__(self, model_path: Union[str, Path] = "models/arrhenius_params.json"):
        self.Ea_R: Optional[float] = None      # Энергия активации / Газовая постоянная
        self.ln_A: Optional[float] = None      # Натуральный логарифм предэкспоненциального множителя
        self.is_fitted: bool = False
        self.model_path = Path(model_path)
        self.data: Optional[pd.DataFrame] = None

    @staticmethod
    def _arrhenius_equation(T_K: np.ndarray, ln_A: float, Ea_R: float) -> np.ndarray:
        """
        Уравнение Аррениуса для ВРЕМЕНИ жизни (t ~ 1/k).
        t = A_time * exp(+Ea / RT)
        Внимание: знак ПЛЮС, так как время обратно пропорционально скорости реакции.
        """
        return np.exp(ln_A + Ea_R / T_K)

    def generate_synthetic_data(self, n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
        """Генерирует физически достоверный датасет."""
        np.random.seed(seed)
        
        temp_c = np.random.uniform(15, 35, n_samples)
        temp_k = temp_c + 273.15
        catalyst_pct = np.random.uniform(0.05, 0.3, n_samples)
        humidity = np.random.uniform(30, 80, n_samples)
        nco_oh_ratio = np.random.uniform(1.0, 1.15, n_samples)

        # Физическое ядро
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

    def fit(self, df: Optional[pd.DataFrame] = None) -> None:
        """Обучает модель, находя параметры Ea/R и ln(A)."""
        if df is None and self.data is None:
            raise ValueError("Нет данных для обучения. Передайте df или сначала вызовите generate_synthetic_data()")
            
        data = df if df is not None else self.data
        T_K = data['Temperature_C'].values + 273.15
        y = data['Pot_Life_min'].values
        
        # Фильтрация некорректных значений
        mask = y > 0
        T_K_clean = T_K[mask]
        y_clean = y[mask]

        try:
            # maxfev увеличен для гарантии сходимости сложных кривых
            popt, pcov = curve_fit(
                self._arrhenius_equation, 
                T_K_clean, 
                y_clean, 
                p0=[-8.0, 3000.0], # Начальные приближения: ln(A) ~ -8, Ea/R ~ 3000
                maxfev=10000
            )
            self.ln_A, self.Ea_R = popt
            self.is_fitted = True
            print(f"Модель обучена успешно!\nНайдено Ea/R: {self.Ea_R:.2f} (должно быть > 0)\nНайдено ln(A): {self.ln_A:.2f}")
        except Exception as e:
            print(f"Ошибка при фиттинге модели: {e}")
            self.is_fitted = False

    def predict(self, temperature_c: Union[float, List[float], np.ndarray, pd.Series]) -> np.ndarray:
        """Предсказывает время жизни при заданной температуре (поддерживает векторизацию)."""
        if not self.is_fitted:
            raise RuntimeError("Сначала необходимо обучить модель методом .fit() или загрузить параметры .load_model()")
            
        t_k = np.asarray(temperature_c) + 273.15
        return self._arrhenius_equation(t_k, self.ln_A, self.Ea_R)

    def save_model(self, path: Optional[Union[str, Path]] = None) -> None:
        """Сохраняет параметры модели в JSON файл."""
        if not self.is_fitted:
            raise RuntimeError("Нельзя сохранить необученную модель.")
        
        save_path = Path(path) if path else self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "Ea_R": float(self.Ea_R),
            "ln_A": float(self.ln_A),
            "is_fitted": True
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=4)
        print(f"Параметры модели сохранены в {save_path}")

    def load_model(self, path: Optional[Union[str, Path]] = None) -> None:
        """Загружает параметры модели из JSON файла."""
        load_path = Path(path) if path else self.model_path
        
        if not load_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {load_path}")
            
        with open(load_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
            
        self.Ea_R = model_data["Ea_R"]
        self.ln_A = model_data["ln_A"]
        self.is_fitted = model_data.get("is_fitted", True)
        print(f"Модель успешно загружена из {load_path}")