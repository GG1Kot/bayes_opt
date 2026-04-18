#!/usr/bin/env python3
"""
Запуск сравнительного эксперимента по байесовской оптимизации.

Тестирует 4 метода учёта ограничений (Penalty, Barrier, Lagrange, CEI)
на наборе задач с ограничениями с размерностями 2, 3, 5.

Автор: Elizaveta Surda
Дата: 2026
"""

from src.utils import experiment
from src.utils import visualization

DIMENSIONS = [2, 3, 5]


def main() -> None:
    """
    Запуск сравнительного эксперимента.

    Конфигурация:
        - Размерности: 2, 3, 5
        - Задачи: Sphere, Rosenbrock, Ackley, Rastrigin, Michalewicz
        - Методы: Penalty, Barrier, Lagrange, CEI
        - Количество запусков: 2
        - Итераций на запуск: 20
    """
    print("=" * 80)
    print("ЗАПУСК ЭКСПЕРИМЕНТА ПО БАЙЕСОВСКОЙ ОПТИМИЗАЦИИ")
    print("=" * 80)

    results = experiment.run_comprehensive_experiment(
        dimensions=DIMENSIONS,
        n_runs=2,
        n_iterations=20,
        n_initial_points_factor=5,
    )

    print(f"\nГотово. Выполнено запусков: {results['n_total']}")
    print("=" * 80)

    all_results = results["results"]
    if not all_results:
        print("Нет результатов для визуализации.")
        return

    # Общий интегральный график: сходимость 4 методов по всем задачам и размерностям
    visualization.plot_integral_convergence(
        all_results,
        save_path="results/plots/integral_convergence.png",
    )

    # Детальные графики: отдельно по каждой задаче (subplot на размерность)
    problems = sorted({r.function_name for r in all_results})
    for problem in problems:
        prob_results = [r for r in all_results if r.function_name == problem]
        visualization.plot_convergence_by_dimension(
            prob_results,
            dimensions=DIMENSIONS,
            problem_name=problem,
        )

    # Сводная таблица
    visualization.save_summary_table(all_results)


if __name__ == "__main__":
    main()
