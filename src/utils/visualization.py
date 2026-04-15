"""
Модуль для визуализации результатов экспериментов.

Содержит функции для построения графиков сходимости,
сравнения методов и сохранения результатов.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

from .types import OptimizationResult


def plot_convergence(
    results: List[OptimizationResult],
    title: str = "Сходимость алгоритма",
    save_path: Optional[str] = None,
    show_std: bool = True
) -> None:
    """
    Построение графика сходимости со стандартными отклонениями.
    
    Аргументы:
        results: список результатов оптимизации
        title: заголовок графика
        save_path: путь для сохранения (опционально)
        show_std: показывать ли доверительный интервал
    """
    if not results:
        print("Нет данных для построения графика")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Агрегируем результаты
    max_len = max(len(r.history_values) for r in results)
    
    # Выравниваем историю
    aligned = []
    for r in results:
        hist = r.history_values.copy()
        while len(hist) < max_len:
            hist.append(hist[-1] if hist else np.inf)
        aligned.append(hist)
    
    aligned = np.array(aligned)
    mean_vals = np.mean(aligned, axis=0)
    std_vals = np.std(aligned, axis=0)
    
    # Рисуем основную линию
    iterations = range(len(mean_vals))
    plt.plot(iterations, mean_vals, 'b-', linewidth=2, label='Среднее значение')
    
    # Добавляем доверительный интервал
    if show_std:
        plt.fill_between(
            iterations,
            mean_vals - std_vals,
            mean_vals + std_vals,
            alpha=0.3,
            color='blue',
            label='±1 стандартное отклонение'
        )
    
    # Отмечаем начальную выборку
    n_initial = results[0].n_initial_points if results else 10
    plt.axvline(x=n_initial, color='red', linestyle='--', alpha=0.7, 
                label=f'Конец начальной выборки ({n_initial})')
    
    plt.xlabel('Итерация', fontsize=12)
    plt.ylabel('Лучшее найденное значение', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Логарифмическая шкала если значения сильно различаются
    if np.max(mean_vals) / (np.min(mean_vals[mean_vals > 0]) + 1e-10) > 1000:
        plt.yscale('log')
        plt.ylabel('Лучшее найденное значение (лог. шкала)', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_all_results(results: List[OptimizationResult], dimensions: List[int]) -> None:
    """
    Построение всех графиков для эксперимента.
    
    Аргументы:
        results: список всех результатов
        dimensions: список размерностей
    """
    if not results:
        print("Нет данных для визуализации")
        return
    
    # Создаем директорию для графиков
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    
    # Группируем по задачам
    problems = set(r.function_name for r in results)
    
    for problem in problems:
        problem_results = [r for r in results if r.function_name == problem]
        
        if not problem_results:
            continue
        
        # График сходимости
        plot_convergence(
            problem_results,
            title=f"Сходимость на задаче {problem}",
            save_path=f"results/plots/convergence_{problem.replace(' ', '_')}.png"
        )
        
        # График финальных результатов (boxplot) по размерностям
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data_by_dim = {}
        for dim in dimensions:
            dim_results = [r for r in problem_results if r.dimension == dim and r.best_feasible]
            if dim_results:
                data_by_dim[f"dim={dim}"] = [r.best_value for r in dim_results]
        
        if data_by_dim:
            ax.boxplot(data_by_dim.values(), labels=data_by_dim.keys())
            ax.set_xlabel('Размерность', fontsize=12)
            ax.set_ylabel('Лучшее значение', fontsize=12)
            ax.set_title(f'Распределение результатов на задаче {problem}', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"results/plots/boxplot_{problem.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
            plt.show()


def plot_bbob_results(results: List[OptimizationResult], dimensions: List[int]) -> None:
    """
    Построение графиков для BBOB результатов.
    
    Аргументы:
        results: список результатов
        dimensions: список размерностей
    """
    if not results:
        print("Нет данных для визуализации BBOB результатов")
        return
    
    Path("results/bbob/plots").mkdir(parents=True, exist_ok=True)
    
    # 1. График сходимости по размерностям
    fig, axes = plt.subplots(1, len(dimensions), figsize=(5*len(dimensions), 5))
    if len(dimensions) == 1:
        axes = [axes]
    
    for idx, dim in enumerate(dimensions):
        dim_results = [r for r in results if r.dimension == dim]
        
        if not dim_results:
            axes[idx].text(0.5, 0.5, f'Нет данных для dim={dim}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            continue
        
        # Усредняем историю
        max_len = max(len(r.history_values) for r in dim_results)
        aligned = []
        
        for r in dim_results:
            hist = r.history_values.copy()
            while len(hist) < max_len:
                hist.append(hist[-1] if hist else np.inf)
            aligned.append(hist)
        
        aligned = np.array(aligned)
        mean_vals = np.mean(aligned, axis=0)
        std_vals = np.std(aligned, axis=0)
        
        # Рисуем
        iterations = range(max_len)
        axes[idx].plot(iterations, mean_vals, 'b-', linewidth=2, label='Среднее')
        axes[idx].fill_between(
            iterations,
            mean_vals - std_vals,
            mean_vals + std_vals,
            alpha=0.3,
            color='blue',
            label='±1σ'
        )
        
        # Отмечаем начальную выборку
        n_initial = dim_results[0].n_initial_points if dim_results else 5*dim
        axes[idx].axvline(x=n_initial, color='red', linestyle='--', alpha=0.7,
                         label=f'Начальная выборка ({n_initial})')
        
        axes[idx].set_xlabel('Итерация', fontsize=11)
        axes[idx].set_ylabel('Лучшее значение', fontsize=11)
        axes[idx].set_title(f'Размерность {dim}', fontsize=12)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        
        # Логарифмическая шкала при необходимости
        if np.max(mean_vals) / (np.min(mean_vals[mean_vals > 0]) + 1e-10) > 100:
            axes[idx].set_yscale('log')
    
    plt.suptitle('Сходимость байесовской оптимизации на BBOB-constrained', fontsize=14)
    plt.tight_layout()
    plt.savefig("results/bbob/plots/convergence_by_dim.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. График успешности по размерностям
    fig, ax = plt.subplots(figsize=(10, 6))
    
    success_rates = []
    dim_labels = []
    
    for dim in dimensions:
        dim_results = [r for r in results if r.dimension == dim]
        if dim_results:
            feasible = sum(1 for r in dim_results if r.best_feasible)
            rate = feasible / len(dim_results) * 100
            success_rates.append(rate)
            dim_labels.append(f'dim={dim}')
    
    if success_rates:
        bars = ax.bar(dim_labels, success_rates, color='green', alpha=0.7)
        ax.set_ylim(0, 105)
        ax.set_ylabel('Успешность (%)', fontsize=12)
        ax.set_xlabel('Размерность', fontsize=12)
        ax.set_title('Доля допустимых решений по размерностям', fontsize=14)
        
        # Добавляем значения на столбцы
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig("results/bbob/plots/success_rates.png", dpi=150, bbox_inches='tight')
        plt.show()


def save_bbob_summary_table(results: List[OptimizationResult]) -> None:
    """
    Сохранение сводной таблицы BBOB результатов.
    
    Аргументы:
        results: список результатов
    """
    if not results:
        print("Нет результатов для сохранения")
        return
    
    from collections import defaultdict
    
    grouped = defaultdict(list)
    for r in results:
        key = (r.function_name, r.dimension)
        grouped[key].append(r)
    
    Path("results/bbob").mkdir(parents=True, exist_ok=True)
    
    with open("results/bbob/detailed_summary.txt", 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ BBOB-CONSTRAINED\n")
        f.write("="*100 + "\n\n")
        
        for (func_name, dim), runs in sorted(grouped.items()):
            feasible = [r for r in runs if r.best_feasible]
            
            f.write(f"\n{func_name}\n")
            f.write("-"*80 + "\n")
            f.write(f"Размерность: {dim}\n")
            f.write(f"Запусков: {len(runs)}\n")
            f.write(f"Допустимых: {len(feasible)} ({len(feasible)/len(runs)*100:.1f}%)\n")
            
            if feasible:
                values = [r.best_value for r in feasible]
                f.write(f"Лучшее значение: {np.min(values):.6f}\n")
                f.write(f"Среднее значение: {np.mean(values):.6f}\n")
                f.write(f"Медиана: {np.median(values):.6f}\n")
                f.write(f"Стд. отклонение: {np.std(values):.6f}\n")
            
            f.write("\n")
    
    print(f"\n📊 Детальная таблица сохранена в results/bbob/detailed_summary.txt")


def save_results_table(results: List[OptimizationResult], filename: str = "results/summary_table.txt") -> None:
    """
    Сохранение таблицы результатов.
    
    Аргументы:
        results: список результатов
        filename: имя файла для сохранения
    """
    Path("results").mkdir(exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'Функция':<35} {'Разм':<6} {'Метод':<15} {'Лучшее значение':<20} {'Допустимо':<10}\n")
        f.write("-"*100 + "\n")
        
        for r in results:
            feasible = "Да" if r.best_feasible else "Нет"
            func_name = r.function_name[:33] if len(r.function_name) > 33 else r.function_name
            f.write(f"{func_name:<35} {r.dimension:<6} {r.method_name:<15} "
                   f"{r.best_value:<20.6f} {feasible:<10}\n")
        
        f.write("\n" + "="*100 + "\n")
        
        feasible_count = sum(1 for r in results if r.best_feasible)
        f.write(f"Всего запусков: {len(results)}\n")
        f.write(f"Успешных: {feasible_count} ({feasible_count/len(results)*100:.1f}%)\n")
    
    print(f" Таблица результатов сохранена в {filename}")