"""
Модуль визуализации результатов экспериментов.

Строит:
    - Интегральный график сходимости всех методов (усреднение по всем задачам и размерностям)
    - Детальные графики сходимости по каждой задаче с subplots по размерностям
    - Сводную таблицу результатов

Автор: Elizaveta Surda
Дата: 2026
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils import types

_METHOD_COLORS: dict[str, str] = {
    "Penalty": "blue",
    "Barrier": "orange",
    "Lagrange": "green",
    "CEI": "red",
}


def _align_history(results: list[types.OptimizationResult]) -> tuple[np.ndarray, np.ndarray]:
    """
    Выравнивание историй сходимости до одинаковой длины и вычисление статистик.

    Аргументы:
        results: список результатов одного метода

    Возвращает:
        mean_vals: среднее по запускам
        std_vals: стандартное отклонение по запускам
    """
    max_len = max(len(r.history_values) for r in results)
    aligned = []
    for r in results:
        hist = r.history_values.copy()
        while len(hist) < max_len:
            hist.append(hist[-1] if hist else float("inf"))
        aligned.append(hist)
    arr = np.array(aligned)
    return np.mean(arr, axis=0), np.std(arr, axis=0)


def plot_integral_convergence(
    results: list[types.OptimizationResult],
    save_path: str | None = None,
) -> None:
    """
    Интегральный график сходимости: 4 метода усреднены по всем задачам и размерностям.

    Показывает общую картину — какой метод лучше сходится в среднем
    независимо от конкретной задачи и размерности.

    Аргументы:
        results: список всех результатов эксперимента
        save_path: путь для сохранения PNG (опционально)
    """
    if not results:
        print("Нет данных для интегрального графика")
        return

    methods = sorted({r.method_name for r in results})

    plt.figure(figsize=(10, 6))

    for method in methods:
        color = _METHOD_COLORS.get(method, "black")
        method_results = [r for r in results if r.method_name == method]
        if not method_results:
            continue

        mean_vals, std_vals = _align_history(method_results)
        iters = range(len(mean_vals))

        plt.plot(iters, mean_vals, linewidth=2.5, label=method, color=color)
        plt.fill_between(iters, mean_vals - std_vals, mean_vals + std_vals,
                         alpha=0.15, color=color)

    n_initial = results[0].n_initial_points
    plt.axvline(x=n_initial, color="gray", linestyle="--", alpha=0.6,
                label=f"Конец начальной выборки ({n_initial})")

    n_problems = len({r.function_name for r in results})
    n_dims = len({r.dimension for r in results})
    plt.xlabel("Итерация", fontsize=12)
    plt.ylabel("Среднее лучшее значение", fontsize=12)
    plt.title(
        f"Интегральная сходимость методов\n"
        f"({n_problems} задач × {n_dims} размерности, усреднение по всем запускам)",
        fontsize=13,
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Интегральный график сохранён в {save_path}")

    plt.show()


def plot_convergence_by_dimension(
    results: list[types.OptimizationResult],
    dimensions: list[int],
    problem_name: str = "",
) -> None:
    """
    Сетка графиков: один subplot на размерность, отдельная линия на каждый метод.

    Аргументы:
        results: результаты для одной задачи
        dimensions: список размерностей
        problem_name: название задачи (для заголовка и имени файла)
    """
    if not results:
        return

    Path("results/plots").mkdir(parents=True, exist_ok=True)
    methods = sorted({r.method_name for r in results})

    fig, axes = plt.subplots(
        1, len(dimensions),
        figsize=(6 * len(dimensions), 5),
        sharey=False,
    )
    if len(dimensions) == 1:
        axes = [axes]

    for idx, dim in enumerate(dimensions):
        ax = axes[idx]
        dim_results = [r for r in results if r.dimension == dim]

        if not dim_results:
            ax.text(0.5, 0.5, f"Нет данных\ndim={dim}",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        for method in methods:
            method_results = [r for r in dim_results if r.method_name == method]
            if not method_results:
                continue

            color = _METHOD_COLORS.get(method, "black")
            mean_vals, std_vals = _align_history(method_results)
            iters = range(len(mean_vals))

            ax.plot(iters, mean_vals, linewidth=2, label=method, color=color)
            ax.fill_between(iters, mean_vals - std_vals, mean_vals + std_vals,
                            alpha=0.15, color=color)

        n_initial = dim_results[0].n_initial_points
        ax.axvline(x=n_initial, color="gray", linestyle="--", alpha=0.6,
                   label=f"Начальная выборка ({n_initial})")
        ax.set_xlabel("Итерация", fontsize=11)
        ax.set_ylabel("Лучшее значение", fontsize=11)
        ax.set_title(f"dim = {dim}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    title = f"Сходимость методов — {problem_name}" if problem_name else "Сходимость методов"
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    safe_name = problem_name.replace(" ", "_").replace("/", "_")
    path = f"results/plots/convergence_{safe_name}.png" if safe_name else \
           "results/plots/convergence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def save_summary_table(
    results: list[types.OptimizationResult],
    filename: str = "results/summary_table.txt",
) -> None:
    """
    Сводная таблица результатов: среднее, медиана, лучшее, успешность.

    Аргументы:
        results: список всех результатов
        filename: путь к файлу (.txt)
    """
    if not results:
        print("Нет результатов для сохранения")
        return

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    grouped: dict = defaultdict(list)
    for r in results:
        grouped[(r.function_name, r.dimension, r.method_name)].append(r)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 110 + "\n")
        f.write("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ\n")
        f.write("=" * 110 + "\n\n")

        header = (
            f"{'Функция':<25} {'Разм':<5} {'Метод':<12} "
            f"{'Среднее':<15} {'Медиана':<15} {'Лучшее':<15} "
            f"{'Успешность':<12} {'Время(с)':<10}"
        )
        f.write(header + "\n")
        f.write("-" * 110 + "\n")

        for (func_name, dim, method), runs in sorted(grouped.items()):
            feasible = [r for r in runs if r.best_feasible]
            success_rate = len(feasible) / len(runs) * 100

            if feasible:
                values = [r.best_value for r in feasible]
                mean_v, median_v, best_v = (
                    float(np.mean(values)),
                    float(np.median(values)),
                    float(np.min(values)),
                )
            else:
                mean_v = median_v = best_v = float("inf")

            mean_time = float(np.mean([r.wall_time for r in runs]))
            name_short = func_name[:23] if len(func_name) > 23 else func_name

            f.write(
                f"{name_short:<25} {dim:<5} {method:<12} "
                f"{mean_v:<15.6f} {median_v:<15.6f} {best_v:<15.6f} "
                f"{success_rate:<11.1f}% {mean_time:<10.2f}\n"
            )

        feasible_total = sum(1 for r in results if r.best_feasible)
        f.write("\n" + "=" * 110 + "\n")
        f.write(f"Всего запусков: {len(results)}\n")
        f.write(
            f"Успешных: {feasible_total} / {len(results)} "
            f"({feasible_total / len(results) * 100:.1f}%)\n"
        )

    print(f"\nСводная таблица сохранена в {filename}")