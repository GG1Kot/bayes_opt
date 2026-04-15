"""Модуль с утилитами для экспериментов."""

from src.utils.experiment import run_bbob_experiment, run_comprehensive_experiment
from src.utils.types import OptimizationResult

__all__ = [
    'run_bbob_experiment',
    'run_comprehensive_experiment',
    'OptimizationResult',
]