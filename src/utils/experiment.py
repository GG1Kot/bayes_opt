"""
Проведение экспериментов на тестовых задачах с ограничениями.

При наличии coco-experiment используется BBOB-constrained набор,
иначе — 5 классических функций из constrained_problems.

Автор: Elizaveta Surda
Дата: 2026
"""

import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.bayesian_optimization import barrier
from src.bayesian_optimization import base
from src.bayesian_optimization import cei
from src.bayesian_optimization import lagrange
from src.bayesian_optimization import penalty
from src.utils import types

try:
    import cocoex  # type: ignore[import-untyped]
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False
    print("Предупреждение: coco-experiment не установлен.")
    print("Для BBOB-эксперимента: uv add coco-experiment")

_DIMENSION_TO_INDEX: dict[int, int] = {2: 1, 3: 2, 5: 3, 10: 4, 20: 5, 40: 6}
_INDEX_TO_DIMENSION: dict[int, int] = {v: k for k, v in _DIMENSION_TO_INDEX.items()}


def _build_methods() -> dict[str, Callable]:
    """
    Словарь фабрик методов учёта ограничений.

    Возвращает:
        словарь {имя_метода: фабричная_функция(constraints)}
    """
    return {
        "Penalty": lambda c: penalty.PenaltyMethod(c, penalty_coeff=100.0),
        "Barrier": lambda c: barrier.BarrierMethod(c, barrier_coeff=1.0),
        "Lagrange": lambda c: lagrange.LagrangeMethod(c, penalty_coeff=10.0),
        "CEI": lambda c: cei.ConstrainedExpectedImprovement(c, xi=0.01),
    }


def run_standard_experiment(
    dimensions: list[int] | None = None,
    n_runs: int = 3,
    n_iterations: int = 30,
    n_initial_points_factor: int = 5,
    save_all_points: bool = True,
) -> dict[str, Any]:
    """
    Эксперимент на 5 стандартных тестовых задачах с ограничениями.

    Аргументы:
        dimensions: список размерностей (по умолчанию [2, 3, 5])
        n_runs: число повторных запусков на каждую конфигурацию
        n_iterations: число итераций байесовской оптимизации
        n_initial_points_factor: множитель для размера начальной LHS-выборки

    Возвращает:
        словарь с ключами 'results' и 'n_total'
    """
    from src.test_problems import constrained_problems  # pylint: disable=import-outside-toplevel

    if dimensions is None:
        dimensions = [2, 3, 5]

    test_problems = constrained_problems.get_test_problems(dimensions)
    methods = _build_methods()
    all_results: list[types.OptimizationResult] = []

    print("\n" + "=" * 80)
    print("ЭКСПЕРИМЕНТ НА СТАНДАРТНЫХ ЗАДАЧАХ")
    print("=" * 80)
    print(f"Задач: {len(test_problems)}, методов: {len(methods)}, "
          f"запусков: {n_runs}, итераций: {n_iterations}")
    print("=" * 80)

    for problem in test_problems:
        problem_name: str = problem["name"]
        dim: int = problem["dimension"]
        problem_func: Callable = problem["function"]
        constraints: list[Callable] = problem["constraints"]
        bounds: np.ndarray = problem["bounds"]
        n_initial = n_initial_points_factor * dim

        print(f"\nЗадача: {problem_name}, dim={dim}, ограничений: {len(constraints)}")

        for method_idx, (method_name, method_creator) in enumerate(methods.items()):
            print(f"  Метод: {method_name}")
            for run_id in range(n_runs):
                # Seed зависит от метода, запуска и размерности —
                # каждая комбинация стартует из уникальных точек
                seed = 42 + run_id * 13 + dim * 100 + method_idx * 7
                handler = method_creator(constraints)

                try:
                    start = time.monotonic()
                    optimizer = base.BayesianOptimizer(
                        objective_function=problem_func,
                        bounds=bounds,
                        constraint_handler=handler,
                        n_initial_points=n_initial,
                        random_state=seed,
                    )
                    best_value, best_point, history = optimizer.optimize(n_iterations)
                    wall_time = time.monotonic() - start
                    is_feasible = bool(handler.is_feasible(best_point.reshape(1, -1))[0])
                    all_points = []
                    all_values = []
                    all_constraints = []
                    if save_all_points and hasattr(optimizer, 'X_sample'):
                        # Если в optimizer хранятся все точки
                        all_points = optimizer.X_sample.tolist()
                        all_values = optimizer.Y_sample.tolist() if hasattr(optimizer, 'Y_sample') else []
                    
                    # Альтернатива: если история точек доступна через атрибут
                    elif hasattr(optimizer, 'history_points'):
                        all_points = optimizer.history_points
                        all_values = history
                    all_results.append(types.OptimizationResult(
                        function_name=problem_name,
                        dimension=dim,
                        method_name=method_name,
                        best_value=best_value,
                        best_point=best_point,
                        best_feasible=is_feasible,
                        n_iterations=n_iterations,
                        n_initial_points=n_initial,
                        history_values=history,
                        history_points=all_points,
                        history_constraints=all_constraints,
                        wall_time=wall_time,
                        converged=len(history) == n_iterations + 1,
                    ))
                    print(f"    [{run_id+1}/{n_runs}] {best_value:.6f} "
                          f"({'допустимо' if is_feasible else 'недопустимо'})")

                except Exception as exc:  # pylint: disable=broad-except
                    print(f"    [{run_id+1}/{n_runs}] Ошибка: {exc}")
                

    # if all_results:
    #     _save_results(all_results, dimensions)
    #     # Обновлённое сохранение с точками
    if all_results and save_all_points:
        _save_results_with_points(all_results, dimensions)

    return {"results": all_results, "n_total": len(all_results)}


def run_comprehensive_experiment(
    dimensions: list[int] | None = None,
    n_runs: int = 3,
    n_iterations: int = 30,
    n_initial_points_factor: int = 5,
) -> dict[str, Any]:
    """
    Запуск эксперимента с автоматическим выбором набора задач.

    При наличии COCO — BBOB-constrained, иначе — стандартные задачи.

    Аргументы:
        dimensions: список размерностей (по умолчанию [2, 3, 5])
        n_runs: число повторных запусков на каждую конфигурацию
        n_iterations: число итераций байесовской оптимизации
        n_initial_points_factor: множитель для размера начальной LHS-выборки

    Возвращает:
        словарь с ключами 'results' и 'n_total'
    """
    if dimensions is None:
        dimensions = [2, 3, 5]

    if COCO_AVAILABLE:
        print("\nИспользуем BBOB-constrained (COCO)")
    else:
        print("\nBBOB недоступен, используем стандартные задачи")

    return run_standard_experiment(dimensions, n_runs, n_iterations, n_initial_points_factor)


def _save_results(
    results: list[types.OptimizationResult],
    dimensions: list[int],
) -> None:
    """
    Сохранение результатов эксперимента в текстовый файл.

    Аргументы:
        results: список результатов
        dimensions: список использованных размерностей
    """
    Path("results").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/experiment_{timestamp}.txt"

    methods = sorted({r.method_name for r in results})

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Размерности: {dimensions}\n")
        f.write(f"Всего запусков: {len(results)}\n\n")

        f.write(f"{'Функция':<20} {'Разм':<6} {'Метод':<12} "
                f"{'Лучшее значение':<18} {'Допустимо':<10} {'Время(с)':<10}\n")
        f.write("-" * 100 + "\n")

        for r in results:
            feasible = "Да" if r.best_feasible else "Нет"
            f.write(f"{r.function_name:<20} {r.dimension:<6} {r.method_name:<12} "
                    f"{r.best_value:<18.6f} {feasible:<10} {r.wall_time:<10.2f}\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("СТАТИСТИКА ПО МЕТОДАМ\n")
        f.write("-" * 100 + "\n")

        for method in methods:
            method_results = [r for r in results if r.method_name == method]
            feasible_results = [r for r in method_results if r.best_feasible]
            f.write(f"\n{method}:\n")
            f.write(f"  Запусков: {len(method_results)}, "
                    f"допустимых: {len(feasible_results)}\n")
            if feasible_results:
                values = [r.best_value for r in feasible_results]
                f.write(f"  Среднее: {np.mean(values):.6f}, "
                        f"медиана: {np.median(values):.6f}, "
                        f"лучшее: {np.min(values):.6f}\n")

    print(f"\nРезультаты сохранены в {filename}")

def _save_results_with_points(
    results: list[types.OptimizationResult],
    dimensions: list[int],
) -> None:
    """
    Сохранение результатов с полной историей точек.
    """
    import json
    import pickle
    
    Path("results").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Текстовый отчёт (как раньше)
    _save_results(results, dimensions)
    
    # 2. JSON для машинного чтения (без больших массивов)
    json_data = [r.to_dict(include_history=False) for r in results]
    with open(f"results/summary_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # 3. NPZ со всеми точками (компактное бинарное хранение)
    points_data = {}
    for i, r in enumerate(results):
        if r.history_points:
            key = f"{r.function_name}_dim{r.dimension}_{r.method_name}_{i}"
            points_data[f"{key}_points"] = r.get_all_points_array()
            points_data[f"{key}_values"] = r.get_all_values_array()
            points_data[f"{key}_feasible"] = r.get_feasible_mask()
    
    if points_data:
        np.savez_compressed(f"results/points_{timestamp}.npz", **points_data)
        print(f"✅ Точки сохранены: results/points_{timestamp}.npz")
    
    # 4. Pickle для полного восстановления (опционально)
    with open(f"results/full_results_{timestamp}.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"✅ Полные результаты: results/full_results_{timestamp}.pkl")
