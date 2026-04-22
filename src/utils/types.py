"""
Определение типов данных для проекта.

Автор: Elizaveta Surda
Дата: 2026
"""

from dataclasses import dataclass, field
from typing import Any

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
        history_points: история ВСЕХ протестированных точек (каждая итерация)
        history_constraints: история значений ограничений для каждой точки
        history_iteration_best: лучшее значение на каждой итерации (для сходимости)
        wall_time: время выполнения в секундах
        converged: True, если алгоритм отработал все итерации
        extra: дополнительные данные (для расширения)
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
    
    # НОВЫЕ ПОЛЯ ДЛЯ ПОЛНОЙ ИСТОРИИ
    history_points: list[np.ndarray] = field(default_factory=list)
    history_constraints: list[list[float]] = field(default_factory=list)
    history_iteration_best: list[float] = field(default_factory=list)
    
    # Мета-информация
    wall_time: float = 0.0
    converged: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    
    def get_all_points_array(self) -> np.ndarray:
        """
        Возвращает все протестированные точки в виде 2D массива.
        
        Возвращает:
            массив формы (n_points, dimension)
        """
        if not self.history_points:
            return np.array([])
        return np.vstack([p.reshape(1, -1) for p in self.history_points])
    
    def get_all_values_array(self) -> np.ndarray:
        """
        Возвращает все значения функции в виде массива.
        
        Возвращает:
            массив формы (n_points,)
        """
        return np.array(self.history_values)
    
    def get_feasible_mask(self) -> np.ndarray:
        """
        Возвращает маску допустимых точек.
        
        Возвращает:
            булев массив формы (n_points,)
        """
        if not self.history_constraints:
            return np.ones(len(self.history_values), dtype=bool)
        # Точка допустима, если все ограничения <= 0
        return np.array([all(c <= 1e-6 for c in constr) 
                        for constr in self.history_constraints])
    
    def get_infeasible_points(self) -> list[np.ndarray]:
        """
        Возвращает список недопустимых точек.
        
        Возвращает:
            список точек, нарушающих ограничения
        """
        mask = self.get_feasible_mask()
        return [p for i, p in enumerate(self.history_points) if not mask[i]]
    
    def get_feasible_points(self) -> list[np.ndarray]:
        """
        Возвращает список допустимых точек.
        
        Возвращает:
            список точек, удовлетворяющих ограничениям
        """
        mask = self.get_feasible_mask()
        return [p for i, p in enumerate(self.history_points) if mask[i]]
    
    def to_dict(self, include_history: bool = False) -> dict:
        """
        Преобразует результат в словарь для сериализации.
        
        Аргументы:
            include_history: включать ли полную историю точек
        
        Возвращает:
            словарь с данными
        """
        result = {
            "function_name": self.function_name,
            "dimension": self.dimension,
            "method_name": self.method_name,
            "best_value": float(self.best_value),
            "best_point": self.best_point.tolist(),
            "best_feasible": self.best_feasible,
            "n_iterations": self.n_iterations,
            "n_initial_points": self.n_initial_points,
            "wall_time": self.wall_time,
            "converged": self.converged,
        }
        
        if include_history:
            result["history_values"] = self.history_values
            result["history_points"] = [p.tolist() for p in self.history_points]
            result["history_constraints"] = self.history_constraints
            result["history_iteration_best"] = self.history_iteration_best
        
        result.update(self.extra)
        
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "OptimizationResult":
        """
        Восстанавливает результат из словаря.
        
        Аргументы:
            data: словарь с данными
        
        Возвращает:
            объект OptimizationResult
        """
        history_points = []
        if "history_points" in data:
            history_points = [np.array(p) for p in data["history_points"]]
        
        return cls(
            function_name=data["function_name"],
            dimension=data["dimension"],
            method_name=data["method_name"],
            best_value=data["best_value"],
            best_point=np.array(data["best_point"]),
            best_feasible=data["best_feasible"],
            n_iterations=data["n_iterations"],
            n_initial_points=data["n_initial_points"],
            history_values=data.get("history_values", []),
            history_points=history_points,
            history_constraints=data.get("history_constraints", []),
            history_iteration_best=data.get("history_iteration_best", []),
            wall_time=data.get("wall_time", 0.0),
            converged=data.get("converged", False),
            extra={k: v for k, v in data.items() 
                   if k not in cls.__dataclass_fields__},
        )


@dataclass
class ExperimentConfig:
    """
    Конфигурация эксперимента.
    
    Атрибуты:
        dimensions: список размерностей
        n_runs: число повторных запусков
        n_iterations: число итераций оптимизации
        n_initial_points_factor: множитель начальной выборки
        methods: список методов для тестирования
        save_all_points: сохранять ли все точки оптимизации
        random_seed: базовый seed для воспроизводимости
    """
    dimensions: list[int] = field(default_factory=lambda: [2, 3, 5])
    n_runs: int = 3
    n_iterations: int = 30
    n_initial_points_factor: int = 5
    methods: list[str] = field(default_factory=lambda: ["Penalty", "Barrier", "Lagrange", "CEI"])
    save_all_points: bool = True
    random_seed: int = 42
    
    def get_n_initial_points(self, dimension: int) -> int:
        """Возвращает размер начальной выборки для заданной размерности."""
        return self.n_initial_points_factor * dimension