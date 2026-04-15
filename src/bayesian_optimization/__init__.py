"""
Модуль с ядром байесовской оптимизации.

Содержит базовый оптимизатор и различные методы обработки ограничений:
- Barrier method (барьерный метод)
- Lagrange method (метод множителей Лагранжа)
- Penalty method (метод штрафных функций)
- CEI (Constrained Expected Improvement)

Пример использования:
    from core import BayesianOptimizer, PenaltyMethod
    
    optimizer = BayesianOptimizer(
        objective_function=func,
        bounds=bounds,
        constraint_handler=PenaltyMethod(penalty_coeff=100)
    )
"""

from .base import BayesianOptimizer
from .barrier import BarrierMethod
from .lagrange import LagrangeMethod
from .penalty import PenaltyMethod
from .cei import ConstrainedExpectedImprovement

__all__ = [
    'BayesianOptimizer',
    'BarrierMethod',
    'LagrangeMethod',
    'PenaltyMethod',
    'ConstrainedExpectedImprovement',
]