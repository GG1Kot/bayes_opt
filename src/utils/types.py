"""
Определение типов данных для проекта.

Содержит dataclass для хранения результатов оптимизации.

Автор: Elizaveta Surda
Дата: 2026
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class OptimizationResult:
    """
    Результаты одного запуска оптимизации.
    
    Атрибуты:
        function_name: название тестовой функции
        dimension: размерность задачи
        method_name: название метода оптимизации
        best_value: лучшее найденное значение функции
        best_point: лучшая найденная точка
        best_feasible: флаг допустимости лучшего решения
        n_iterations: количество итераций
        n_initial_points: размер начальной выборки
        history_values: история лучших значений по итерациям
        wall_time: время выполнения в секундах
        converged: флаг сходимости
    """
    function_name: str
    dimension: int
    method_name: str
    best_value: float
    best_point: np.ndarray
    best_feasible: bool
    n_iterations: int
    n_initial_points: int
    history_values: List[float] = field(default_factory=list)
    wall_time: float = 0.0
    converged: bool = False