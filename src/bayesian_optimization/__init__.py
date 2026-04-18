"""
Ядро байесовской оптимизации с ограничениями.

Методы учёта ограничений:
    barrier: барьерный метод (логарифмические барьеры)
    lagrange: Augmented Lagrangian
    penalty: метод штрафных функций
    cei: Constrained Expected Improvement
"""

from src.bayesian_optimization import barrier
from src.bayesian_optimization import base
from src.bayesian_optimization import cei
from src.bayesian_optimization import lagrange
from src.bayesian_optimization import penalty

__all__ = ["barrier", "base", "cei", "lagrange", "penalty"]
