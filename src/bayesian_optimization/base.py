"""
Базовый класс байесовского оптимизатора.

Реализует основной цикл оптимизации с использованием Gaussian Process
в качестве суррогатной модели. Поддерживает различные методы обработки
ограничений через механизм constraint_handler.

Автор: Elizaveta Surda
Дата: 2026
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from typing import Callable, Optional, List, Tuple
from abc import ABC, abstractmethod
import warnings

from src.experimental_design.lhs import latin_hypercube_sample

warnings.filterwarnings("ignore")


class ConstraintHandler(ABC):
    """
    Абстрактный базовый класс для обработчиков ограничений.
    
    Определяет интерфейс, который должны реализовать все методы
    учета ограничений: PenaltyMethod, BarrierMethod, LagrangeMethod, CEI.
    
    Методы:
        evaluate_constraints: оценка нарушения ограничений
        compute_penalized_objective: вычисление штрафованной целевой функции
        get_acquisition_weights: получение весов для acquisition функции
        is_feasible: проверка допустимости точек
    """
    
    @abstractmethod
    def evaluate_constraints(self, X: np.ndarray) -> np.ndarray:
        """
        Оценка нарушения ограничений.
        
        Параметры:
            X: точки для оценки, форма (n_points, n_dims)
            
        Возвращает:
            violations: массив нарушений, форма (n_points,)
                        0 - точка допустима, >0 - нарушение
        """
        pass
    
    @abstractmethod
    def compute_penalized_objective(self, X: np.ndarray, f_values: np.ndarray) -> np.ndarray:
        """
        Вычисление штрафованной целевой функции.
        
        Параметры:
            X: точки, форма (n_points, n_dims)
            f_values: значения целевой функции, форма (n_points,)
            
        Возвращает:
            penalized: значения штрафованной функции, форма (n_points,)
        """
        pass
    
    @abstractmethod
    def get_acquisition_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Получение весов для acquisition функции.
        
        Параметры:
            X: точки для оценки, форма (n_points, n_dims)
            
        Возвращает:
            weights: веса в диапазоне [0, 1], форма (n_points,)
        """
        pass
    
    @abstractmethod
    def is_feasible(self, X: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        """
        Проверка допустимости точек.
        
        Параметры:
            X: точки для проверки, форма (n_points, n_dims)
            tolerance: допуск на нарушение ограничений
            
        Возвращает:
            feasible: булев массив, True если точка допустима
        """
        pass


class BayesianOptimizer:
    """
    Байесовский оптимизатор с поддержкой ограничений.
    
    Использует Gaussian Process для моделирования целевой функции
    и Expected Improvement для выбора следующей точки.
    
    Алгоритм:
        1. Генерация начальной выборки методом LHS
        2. Построение GP модели на всех данных
        3. Максимизация acquisition функции
        4. Оценка целевой функции в новой точке
        5. Повторение шагов 2-4
    
    Параметры:
        objective_function: целевая функция f(x) -> float
        bounds: границы переменных, форма (n_dims, 2)
        constraint_handler: обработчик ограничений
        n_initial_points: размер начальной выборки
        random_state: seed для воспроизводимости
    
    Пример:
        >>> bounds = np.array([[-5, 5], [-5, 5]])
        >>> def sphere(x): return x[0]**2 + x[1]**2
        >>> optimizer = BayesianOptimizer(sphere, bounds)
        >>> best_val, best_point, history = optimizer.optimize(50)
    """
    
    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        constraint_handler: Optional[ConstraintHandler] = None,
        n_initial_points: int = 10,
        random_state: int = 42
    ):
        """
        Инициализация оптимизатора.
        
        Параметры:
            objective_function: целевая функция
            bounds: границы переменных
            constraint_handler: обработчик ограничений
            n_initial_points: размер начальной выборки
            random_state: seed для воспроизводимости
        """
        self.objective = objective_function
        self.bounds = np.array(bounds, dtype=float)
        self.n_dims = self.bounds.shape[0]
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        self.constraint_handler = constraint_handler
        
        # Инициализация Gaussian Process
        kernel = Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=random_state
        )
        
        # Хранилище данных
        self.X = None          # Оцененные точки
        self.y = None          # Значения целевой функции
        self.penalized_y = None # Штрафованные значения
        
        np.random.seed(random_state)
    
    def _initial_sample(self) -> np.ndarray:
        """
        Создание начальной выборки.
        
        Используется метод латинского гиперкуба (LHS) для равномерного
        покрытия пространства поиска.
        
        Возвращает:
            X_init: начальные точки, форма (n_initial_points, n_dims)
        """
        return latin_hypercube_sample(self.bounds, self.n_initial_points, self.random_state)
    
    def _update_model(self) -> None:
        """Обновление GP модели на основе всех накопленных данных."""
        if self.X is not None and len(self.X) > self.n_initial_points:
            finite_mask = np.isfinite(self.penalized_y)
            if np.sum(finite_mask) > self.n_initial_points:
                self.gp.fit(self.X[finite_mask], self.penalized_y[finite_mask])
    
    def _acquisition_function(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """
        Расчет Expected Improvement (EI) acquisition функции.
        
        EI(x) = E[max(f_best - f(x), 0)]
        
        Параметры:
            X: точки для расчета, форма (n_points, n_dims)
            xi: параметр exploration (trade-off)
            
        Возвращает:
            ei: значения EI, форма (n_points,)
        """
        X = np.atleast_2d(X)
        
        finite_mask = np.isfinite(self.penalized_y)
        if np.sum(finite_mask) > 0:
            mu, sigma = self.gp.predict(X, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            f_best = np.min(self.penalized_y[finite_mask])
        else:
            mu = np.zeros(len(X))
            sigma = np.ones(len(X))
            f_best = 0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = (f_best - mu - xi) / sigma
            ei = (f_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0] = 0
            ei[ei < 0] = 0
        
        # Если есть обработчик ограничений, применяем веса
        if self.constraint_handler:
            weights = self.constraint_handler.get_acquisition_weights(X)
            return ei * weights
        
        return ei
    
    def _next_point(self, n_restarts: int = 5) -> np.ndarray:
        """
        Поиск следующей точки для оценки.
        
        Использует L-BFGS-B с несколькими случайными рестартами
        для глобальной оптимизации acquisition функции.
        
        Параметры:
            n_restarts: количество случайных рестартов
            
        Возвращает:
            x_next: следующая точка для оценки
        """
        best_x = None
        best_acq = -np.inf
        
        for _ in range(n_restarts):
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            try:
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1)),
                    x0,
                    bounds=self.bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 100}
                )
                if result.success:
                    acq_value = -result.fun
                    if acq_value > best_acq:
                        best_acq = acq_value
                        best_x = result.x
            except Exception:
                continue
        
        return best_x if best_x is not None else x0
    
    def optimize(self, n_iterations: int, verbose: bool = False) -> Tuple[float, np.ndarray, List[float]]:
        """
        Запуск основного цикла оптимизации.
        
        Параметры:
            n_iterations: количество итераций оптимизации
            verbose: печатать ли прогресс
            
        Возвращает:
            best_value: лучшее найденное значение функции
            best_point: лучшая найденная точка
            history: история лучших значений по итерациям
        """
        # Шаг 1: Начальная выборка
        self.X = self._initial_sample()
        self.y = np.array([self.objective(x) for x in self.X])
        
        # Шаг 2: Вычисление штрафованных значений
        if self.constraint_handler:
            self.penalized_y = self.constraint_handler.compute_penalized_objective(self.X, self.y)
            feasible_mask = self.constraint_handler.is_feasible(self.X)
        else:
            self.penalized_y = self.y.copy()
            feasible_mask = np.ones(len(self.X), dtype=bool)
        
        # Шаг 3: Поиск лучшего начального решения
        if np.any(feasible_mask):
            best_idx = np.argmin(self.y[feasible_mask])
            best_value = self.y[feasible_mask][best_idx]
            best_point = self.X[feasible_mask][best_idx].copy()
        else:
            best_idx = np.argmin(self.penalized_y)
            best_value = self.y[best_idx]
            best_point = self.X[best_idx].copy()
        
        history = [best_value]
        
        # Шаг 4: Основной цикл оптимизации
        for iteration in range(n_iterations):
            self._update_model()
            next_x = self._next_point()
            
            if next_x is None:
                break
            
            next_y = self.objective(next_x)
            
            self.X = np.vstack([self.X, next_x.reshape(1, -1)])
            self.y = np.append(self.y, next_y)
            
            if self.constraint_handler:
                penalized = self.constraint_handler.compute_penalized_objective(
                    next_x.reshape(1, -1), np.array([next_y])
                )
                self.penalized_y = np.append(self.penalized_y, penalized[0])
                feasible_mask = self.constraint_handler.is_feasible(self.X)
            else:
                self.penalized_y = np.append(self.penalized_y, next_y)
                feasible_mask = np.ones(len(self.X), dtype=bool)
            
            if np.any(feasible_mask):
                current_best_idx = np.argmin(self.y[feasible_mask])
                current_best_value = self.y[feasible_mask][current_best_idx]
                if current_best_value < best_value:
                    best_value = current_best_value
                    best_point = self.X[feasible_mask][current_best_idx].copy()
            
            history.append(best_value)
        
        return best_value, best_point, history