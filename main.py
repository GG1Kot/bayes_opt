#!/usr/bin/env python3
"""
Главный модуль для запуска эксперимента по байесовской оптимизации.

Запускает сравнительное тестирование 4 методов (Penalty, Barrier, Lagrange, CEI)
на BBOB-constrained задачах (первые 6 функций Sphere) с размерностями 2,3,5.

Автор: Elizaveta Surda
Дата: 2026
"""

import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.experiment import run_comprehensive_experiment


def main():
    """
    Запуск эксперимента.
    
    Конфигурация:
        - Размерности: 2, 3, 5 (доступные в BBOB)
        - Задачи: Sphere (6 вариантов ограничений)
        - Методы: Penalty, Barrier, Lagrange, CEI
        - Количество запусков: 2
        - Итераций: 20
    """
    print("="*80)
    print("ЗАПУСК ЭКСПЕРИМЕНТА ПО БАЙЕСОВСКОЙ ОПТИМИЗАЦИИ")
    print("="*80)
    
    results = run_comprehensive_experiment(
        dimensions=[2, 3, 5],
        n_runs=2,
        n_iterations=20,
        n_initial_points_factor=5
    )
    
    print(f"\nГотово. Выполнено запусков: {results['n_total']}")
    print("="*80)


if __name__ == "__main__":
    main()