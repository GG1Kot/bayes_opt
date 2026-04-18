"""
Определение типов данных для проекта.

Автор: Elizaveta Surda
Дата: 2026
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptimizationResult:
    """
    Результаты одного запуска оптимизации.

    Атрибуты:
        function_name: название тестовой функции
        dimension: размерность задачи
        method_name: название метода учёта ограничений
        best_value: лучшее найденное значение целевой функции
        best_point: координаты лучшей найденной точки
        best_feasible: True, если лучшая точка допустима
        n_iterations: количество итераций оптимизации
        n_initial_points: размер начальной LHS-выборки
        history_values: история лучших значений по итерациям
        wall_time: время выполнения в секундах
        converged: True, если алгоритм отработал все итерации
    """

    function_name: str
    dimension: int
    method_name: str
    best_value: float
    best_point: np.ndarray
    best_feasible: bool
    n_iterations: int
    n_initial_points: int
    history_values: list[float] = field(default_factory=list)
    wall_time: float = 0.0
    converged: bool = False
