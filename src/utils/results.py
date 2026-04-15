"""
Модуль для работы с результатами экспериментов.

Содержит функции для сохранения и печати результатов.
"""

import numpy as np
from typing import List, Dict, Any
from pathlib import Path

from .types import OptimizationResult


def save_results_table(results: List[OptimizationResult], filename: str = None) -> None:
    """
    Сохранение таблицы результатов в текстовый файл.
    
    Аргументы:
        results: список результатов оптимизации
        filename: имя файла для сохранения (опционально)
    """
    if filename is None:
        filename = "results/summary_table.txt"
    
    Path("results").mkdir(exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'Задача':<30} {'Разм':<5} {'Метод':<15} {'Лучшее значение':<20} {'Допустимо':<10} {'Время (с)':<10}\n")
        f.write("-"*100 + "\n")
        
        for r in results:
            feasible = "Да" if r.best_feasible else "Нет"
            # Обрезаем длинные названия
            func_name = r.function_name[:28] if len(r.function_name) > 28 else r.function_name
            f.write(f"{func_name:<30} {r.dimension:<5} {r.method_name:<15} "
                   f"{r.best_value:<20.6f} {feasible:<10} {r.wall_time:<10.2f}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write(f"Всего запусков: {len(results)}\n")
        
        feasible_count = sum(1 for r in results if r.best_feasible)
        f.write(f"Успешных (допустимых решений): {feasible_count}\n")
        f.write(f"Процент успеха: {feasible_count/len(results)*100:.1f}%\n")
    
    print(f"\n Сводная таблица сохранена в {filename}")


def print_summary(results: List[OptimizationResult]) -> None:
    """
    Вывод краткой сводки результатов в консоль.
    
    Аргументы:
        results: список результатов оптимизации
    """
    if not results:
        print("Нет результатов для отображения")
        return
    
    # Группировка по функциям и размерностям
    from collections import defaultdict
    grouped = defaultdict(list)
    
    for r in results:
        key = (r.function_name, r.dimension, r.method_name)
        grouped[key].append(r)
    
    print("\n" + "="*80)
    print("КРАТКАЯ СВОДКА РЕЗУЛЬТАТОВ")
    print("="*80)
    
    for (func_name, dim, method), runs in sorted(grouped.items()):
        feasible_runs = [r for r in runs if r.best_feasible]
        success_rate = len(feasible_runs) / len(runs) * 100
        
        if feasible_runs:
            best_values = [r.best_value for r in feasible_runs]
            mean_best = np.mean(best_values)
            std_best = np.std(best_values)
            median_best = np.median(best_values)
        else:
            mean_best = np.inf
            std_best = 0
            median_best = np.inf
        
        print(f"\n{func_name} (dim={dim}, метод={method})")
        print(f"  Успешность: {success_rate:.1f}% ({len(feasible_runs)}/{len(runs)})")
        
        if feasible_runs:
            print(f"  Среднее лучшее: {mean_best:.6f} ± {std_best:.6f}")
            print(f"  Медиана: {median_best:.6f}")
            print(f"  Лучшее: {np.min(best_values):.6f}")
            print(f"  Худшее: {np.max(best_values):.6f}")
        else:
            print("  ⚠️ Нет допустимых решений")
        
        print(f"  Среднее время: {np.mean([r.wall_time for r in runs]):.2f} сек")


def aggregate_results(results: List[OptimizationResult]) -> Dict[str, Any]:
    """
    Агрегация результатов по всем запускам.
    
    Аргументы:
        results: список результатов
        
    Returns:
        словарь с агрегированными статистиками
    """
    if not results:
        return {}
    
    feasible_results = [r for r in results if r.best_feasible]
    
    # Вычисление статистик
    best_values = [r.best_value for r in feasible_results] if feasible_results else []
    
    # История сходимости (усредненная)
    max_len = max(len(r.history_values) for r in results) if results else 0
    
    history_mean = []
    history_std = []
    
    if max_len > 0:
        # Выравнивание истории
        aligned = []
        for r in results:
            hist = r.history_values.copy()
            while len(hist) < max_len:
                hist.append(hist[-1] if hist else np.inf)
            aligned.append(hist)
        
        aligned = np.array(aligned)
        history_mean = np.mean(aligned, axis=0).tolist()
        history_std = np.std(aligned, axis=0).tolist()
    
    return {
        'n_total': len(results),
        'n_feasible': len(feasible_results),
        'success_rate': len(feasible_results) / len(results) * 100 if results else 0,
        'mean_best': np.mean(best_values) if best_values else np.inf,
        'std_best': np.std(best_values) if best_values else 0,
        'median_best': np.median(best_values) if best_values else np.inf,
        'min_best': np.min(best_values) if best_values else np.inf,
        'max_best': np.max(best_values) if best_values else np.inf,
        'mean_time': np.mean([r.wall_time for r in results]) if results else 0,
        'history_mean': history_mean,
        'history_std': history_std,
    }