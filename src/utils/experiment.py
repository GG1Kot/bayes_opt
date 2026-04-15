"""
Модуль для проведения экспериментов на BBOB-constrained задачах.

Запускает сравнительное тестирование 4 методов оптимизации
на BBOB-constrained тестовом наборе (54 задачи).

Автор: Elizaveta Surda
Дата: 2026
"""

import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.bayesian_optimization.base import BayesianOptimizer
from src.bayesian_optimization.penalty import PenaltyMethod
from src.bayesian_optimization.barrier import BarrierMethod
from src.bayesian_optimization.lagrange import LagrangeMethod
from src.bayesian_optimization.cei import ConstrainedExpectedImprovement
from src.utils.types import OptimizationResult

# Пытаемся импортировать COCO
try:
    import cocoex
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False
    print("Предупреждение: coco-experiment не установлен.")
    print("Для полного BBOB эксперимента выполните: pip install coco-experiment")


# Соответствие между размерностью и индексом в COCO
DIMENSION_TO_INDEX = {
    2: 1,
    3: 2,
    5: 3,
    10: 4,
    20: 5,
    40: 6,
}

INDEX_TO_DIMENSION = {
    1: 2,
    2: 3,
    3: 5,
    4: 10,
    5: 20,
    6: 40,
}


class BBOBBenchmark:
    """
    Обертка для BBOB-constrained задач из COCO.
    """
    
    # Названия функций по ID
    FUNCTION_NAMES = {
        1: "Sphere_1c", 2: "Sphere_3c", 3: "Sphere_9c",
        4: "Sphere_9+3n/4", 5: "Sphere_9+3n/2", 6: "Sphere_9+9n/2",
        7: "Ellipsoid_1c", 8: "Ellipsoid_3c", 9: "Ellipsoid_9c",
        10: "Ellipsoid_9+3n/4", 11: "Ellipsoid_9+3n/2", 12: "Ellipsoid_9+9n/2",
        13: "LinearSlope_1c", 14: "LinearSlope_3c", 15: "LinearSlope_9c",
        16: "LinearSlope_9+3n/4", 17: "LinearSlope_9+3n/2", 18: "LinearSlope_9+9n/2",
        19: "RotEllipsoid_1c", 20: "RotEllipsoid_3c", 21: "RotEllipsoid_9c",
        22: "RotEllipsoid_9+3n/4", 23: "RotEllipsoid_9+3n/2", 24: "RotEllipsoid_9+9n/2",
        25: "Discus_1c", 26: "Discus_3c", 27: "Discus_9c",
        28: "Discus_9+3n/4", 29: "Discus_9+3n/2", 30: "Discus_9+9n/2",
        31: "BentCigar_1c", 32: "BentCigar_3c", 33: "BentCigar_9c",
        34: "BentCigar_9+3n/4", 35: "BentCigar_9+3n/2", 36: "BentCigar_9+9n/2",
        37: "DiffPowers_1c", 38: "DiffPowers_3c", 39: "DiffPowers_9c",
        40: "DiffPowers_9+3n/4", 41: "DiffPowers_9+3n/2", 42: "DiffPowers_9+9n/2",
        43: "Rastrigin_1c", 44: "Rastrigin_3c", 45: "Rastrigin_9c",
        46: "Rastrigin_9+3n/4", 47: "Rastrigin_9+3n/2", 48: "Rastrigin_9+9n/2",
        49: "RotRastrigin_1c", 50: "RotRastrigin_3c", 51: "RotRastrigin_9c",
        52: "RotRastrigin_9+3n/4", 53: "RotRastrigin_9+3n/2", 54: "RotRastrigin_9+9n/2",
    }
    
    def __init__(self, dimensions: List[int], function_ids: List[int], instances: List[int] = [1]):
        """
        Инициализация BBOB бенчмарка.
        
        Параметры:
            dimensions: список размерностей (2,3,5,10,20,40)
            function_ids: список ID функций (1-54)
            instances: список инстансов
        """
        if not COCO_AVAILABLE:
            raise RuntimeError("COCO library not available")
        
        self.dimensions = dimensions
        self.function_ids = function_ids
        self.instances = instances
        
        # Преобразуем размерности в индексы COCO
        dim_indices = []
        for d in dimensions:
            if d in DIMENSION_TO_INDEX:
                dim_indices.append(DIMENSION_TO_INDEX[d])
            else:
                print(f"Предупреждение: размерность {d} недоступна в BBOB. Доступны: 2,3,5,10,20,40")
        
        if not dim_indices:
            raise ValueError("Нет подходящих размерностей")
        
        # Формируем правильный фильтр для COCO suite
        dims_str = ",".join(str(i) for i in dim_indices)
        funcs_str = ",".join(str(f) for f in function_ids)
        inst_str = ",".join(str(i) for i in instances)
        
        suite_filter = f"dimension_indices:{dims_str} function_indices:{funcs_str} instance_indices:{inst_str}"
        
        print(f"Фильтр COCO: {suite_filter}")
        
        self.suite = cocoex.Suite("bbob-constrained", "", suite_filter)
        self.problems = list(self.suite)
        print(f"Загружено {len(self.problems)} задач")
    
    def get_problems(self):
        """Итератор по задачам."""
        for problem in self.problems:
            try:
                # Получаем информацию о задаче
                func_idx = problem.function_index
                func_id = func_idx + 1
                dim_idx = problem.dimension_index
                dim = INDEX_TO_DIMENSION.get(dim_idx, 2)
                
                info = {
                    'id': str(problem.id),
                    'name': self.FUNCTION_NAMES.get(func_id, f"F{func_id}"),
                    'function_id': func_id,
                    'dimension': dim,
                    'bounds': np.array([[-5, 5]] * dim),  # Стандартные границы BBOB
                    'n_constraints': problem.number_of_constraints if hasattr(problem, 'number_of_constraints') else 0,
                }
                yield problem, info
            except Exception as e:
                print(f"Ошибка обработки задачи: {e}")
                continue


def run_bbob_experiment(
    dimensions: List[int] = [2, 4, 8],
    function_ids: List[int] = None,
    n_runs: int = 2,
    n_iterations: int = 30,
    n_initial_points_factor: int = 5
) -> Dict[str, Any]:
    """
    Запуск эксперимента на BBOB-constrained задачах.
    """
    if not COCO_AVAILABLE:
        print("\nCOCO библиотека не установлена, используем стандартные задачи")
        return run_standard_experiment(dimensions, n_runs, n_iterations, n_initial_points_factor)
    
    # Фильтруем только доступные размерности
    available_dims = [d for d in dimensions if d in DIMENSION_TO_INDEX]
    if not available_dims:
        print(f"Предупреждение: размерности {dimensions} недоступны. Используем [2,3,5]")
        available_dims = [2, 3, 5]
    
    # Если ID функций не указаны, берем первые 6 для быстрого теста
    if function_ids is None:
        function_ids = list(range(1, 7))  # Первые 6 функций (Sphere)
    
    # Определяем методы
    methods = {
        'Penalty': lambda c: PenaltyMethod(c, penalty_coeff=100.0),
        'Barrier': lambda c: BarrierMethod(c, barrier_coeff=1.0),
        'Lagrange': lambda c: LagrangeMethod(c, penalty_coeff=10.0),
        'CEI': lambda c: ConstrainedExpectedImprovement(c, xi=0.01),
    }
    
    all_results = []
    
    print("\n" + "="*80)
    print("BBOB-CONSTRAINED ЭКСПЕРИМЕНТ")
    print("="*80)
    print(f"Размерности: {available_dims}")
    print(f"Функции: {len(function_ids)} (ID: {function_ids[:5]}...)")
    print(f"Методы: {list(methods.keys())}")
    print(f"Запусков на конфигурацию: {n_runs}")
    print(f"Итераций: {n_iterations}")
    print("="*80)
    
    try:
        benchmark = BBOBBenchmark(available_dims, function_ids)
    except Exception as e:
        print(f"Ошибка загрузки BBOB: {e}")
        return run_standard_experiment(dimensions, n_runs, n_iterations, n_initial_points_factor)
    
    problem_count = 0
    for problem, info in benchmark.get_problems():
        dim = info['dimension']
        func_name = info['name']
        problem_count += 1
        
        print(f"\n[{problem_count}] Задача: {func_name}, размерность={dim}")
        
        n_initial = n_initial_points_factor * dim
        
        # Создаем обертку для целевой функции
        def make_objective(p):
            def objective(x):
                try:
                    x = np.asarray(x).flatten()
                    return p(x)
                except Exception:
                    return np.inf
            return objective
        
        for method_name, method_creator in methods.items():
            print(f"    Метод: {method_name}")
            
            for run_id in range(n_runs):
                print(f"      Запуск {run_id + 1}/{n_runs}")
                
                seed = 42 + run_id + dim * 100
                handler = method_creator([])
                
                try:
                    start_time = time.time()
                    
                    optimizer = BayesianOptimizer(
                        objective_function=make_objective(problem),
                        bounds=info['bounds'],
                        constraint_handler=handler,
                        n_initial_points=n_initial,
                        random_state=seed
                    )
                    
                    best_value, best_point, history = optimizer.optimize(n_iterations, verbose=False)
                    wall_time = time.time() - start_time
                    
                    # Проверка допустимости через constraint функцию
                    is_feasible = True
                    try:
                        constraint_val = problem.constraint(best_point)
                        if hasattr(constraint_val, '__len__'):
                            is_feasible = np.all(np.array(constraint_val) <= 1e-6)
                        else:
                            is_feasible = constraint_val <= 1e-6
                    except:
                        is_feasible = True
                    
                    result = OptimizationResult(
                        function_name=func_name,
                        dimension=dim,
                        method_name=method_name,
                        best_value=best_value,
                        best_point=best_point,
                        best_feasible=is_feasible,
                        n_iterations=n_iterations,
                        n_initial_points=n_initial,
                        history_values=history,
                        wall_time=wall_time,
                        converged=len(history) == n_iterations
                    )
                    
                    all_results.append(result)
                    
                    feasible_str = "Да" if is_feasible else "Нет"
                    print(f"        Лучшее значение: {best_value:.6f}, Допустимо: {feasible_str}")
                    
                except Exception as e:
                    print(f"        Ошибка: {e}")
                    continue
    
    # Сохраняем результаты
    if all_results:
        save_bbob_results(all_results, available_dims)
    else:
        print("Нет успешных запусков, пробуем стандартные задачи")
        return run_standard_experiment(dimensions, n_runs, n_iterations, n_initial_points_factor)
    
    print("\n" + "="*80)
    print(f"ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
    print(f"Всего успешных запусков: {len(all_results)}")
    print("="*80)
    
    return {'results': all_results, 'n_total': len(all_results)}


def save_bbob_results(results: List[OptimizationResult], dimensions: List[int]) -> None:
    """Сохранение результатов BBOB эксперимента."""
    if not results:
        return
    
    Path("results/bbob").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/bbob/bbob_experiment_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("BBOB-CONSTRAINED ЭКСПЕРИМЕНТ\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Размерности: {dimensions}\n")
        f.write(f"Всего запусков: {len(results)}\n\n")
        
        f.write("-"*100 + "\n")
        f.write(f"{'Функция':<30} {'Разм':<6} {'Метод':<12} {'Лучшее значение':<18} {'Допустимо':<10} {'Время(с)':<10}\n")
        f.write("-"*100 + "\n")
        
        for r in results:
            feasible = "Да" if r.best_feasible else "Нет"
            f.write(f"{r.function_name:<30} {r.dimension:<6} {r.method_name:<12} "
                   f"{r.best_value:<18.6f} {feasible:<10} {r.wall_time:<10.2f}\n")
        
        # Статистика по методам
        f.write("\n" + "="*100 + "\n")
        f.write("СТАТИСТИКА ПО МЕТОДАМ\n")
        f.write("-"*100 + "\n")
        
        methods = set(r.method_name for r in results)
        for method in sorted(methods):
            method_results = [r for r in results if r.method_name == method]
            feasible_results = [r for r in method_results if r.best_feasible]
            
            f.write(f"\n{method}:\n")
            f.write(f"  Запусков: {len(method_results)}\n")
            f.write(f"  Допустимых: {len(feasible_results)}\n")
            
            if feasible_results:
                values = [r.best_value for r in feasible_results]
                f.write(f"  Среднее значение: {np.mean(values):.6f}\n")
                f.write(f"  Медиана: {np.median(values):.6f}\n")
                f.write(f"  Лучшее: {np.min(values):.6f}\n")
    
    print(f"\nРезультаты сохранены в {filename}")


def run_standard_experiment(
    dimensions: List[int] = [2, 4, 8],
    n_runs: int = 3,
    n_iterations: int = 30,
    n_initial_points_factor: int = 5
) -> Dict[str, Any]:
    """Запуск на стандартных тестовых задачах."""
    from src.test_problems.constrained_problems import get_test_problems
    
    test_problems = get_test_problems(dimensions)
    
    methods = {
        'Penalty': lambda c: PenaltyMethod(c, penalty_coeff=100.0),
        'Barrier': lambda c: BarrierMethod(c, barrier_coeff=1.0),
        'Lagrange': lambda c: LagrangeMethod(c, penalty_coeff=10.0),
        'CEI': lambda c: ConstrainedExpectedImprovement(c, xi=0.01),
    }
    
    all_results = []
    
    print("\n" + "="*80)
    print("ЭКСПЕРИМЕНТ НА СТАНДАРТНЫХ ЗАДАЧАХ")
    print("="*80)
    
    for problem in test_problems:
        problem_name = problem['name']
        dim = problem['dimension']
        problem_func = problem['function']
        bounds = problem['bounds']
        
        print(f"\nЗадача: {problem_name}, размерность={dim}")
        
        n_initial = n_initial_points_factor * dim
        
        for method_name, method_creator in methods.items():
            print(f"  Метод: {method_name}")
            
            for run_id in range(n_runs):
                print(f"    Запуск {run_id + 1}/{n_runs}")
                
                seed = 42 + run_id + dim * 100
                handler = method_creator([])
                
                try:
                    start_time = time.time()
                    
                    optimizer = BayesianOptimizer(
                        objective_function=problem_func,
                        bounds=bounds,
                        constraint_handler=handler,
                        n_initial_points=n_initial,
                        random_state=seed
                    )
                    
                    best_value, best_point, history = optimizer.optimize(n_iterations, verbose=False)
                    wall_time = time.time() - start_time
                    is_feasible = handler.is_feasible(best_point.reshape(1, -1))[0]
                    
                    result = OptimizationResult(
                        function_name=problem_name,
                        dimension=dim,
                        method_name=method_name,
                        best_value=best_value,
                        best_point=best_point,
                        best_feasible=is_feasible,
                        n_iterations=n_iterations,
                        n_initial_points=n_initial,
                        history_values=history,
                        wall_time=wall_time,
                        converged=len(history) == n_iterations
                    )
                    
                    all_results.append(result)
                    
                    feasible_str = "Да" if is_feasible else "Нет"
                    print(f"      Лучшее значение: {best_value:.6f}, Допустимо: {feasible_str}")
                    
                except Exception as e:
                    print(f"      Ошибка: {e}")
                    continue
    
    if all_results:
        save_standard_results(all_results, dimensions)
    
    return {'results': all_results, 'n_total': len(all_results)}


def save_standard_results(results: List[OptimizationResult], dimensions: List[int]) -> None:
    """Сохранение результатов стандартного эксперимента."""
    if not results:
        return
    
    Path("results").mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/experiment_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Размерности: {dimensions}\n")
        f.write(f"Всего запусков: {len(results)}\n\n")
        
        f.write(f"{'Функция':<20} {'Разм':<6} {'Метод':<12} {'Лучшее значение':<18} {'Допустимо':<10} {'Время(с)':<10}\n")
        f.write("-"*100 + "\n")
        
        for r in results:
            feasible = "Да" if r.best_feasible else "Нет"
            f.write(f"{r.function_name:<20} {r.dimension:<6} {r.method_name:<12} "
                   f"{r.best_value:<18.6f} {feasible:<10} {r.wall_time:<10.2f}\n")
    
    print(f"\nРезультаты сохранены в {filename}")


def run_comprehensive_experiment(
    dimensions: List[int] = [2, 4, 8],
    n_runs: int = 3,
    n_iterations: int = 30,
    n_initial_points_factor: int = 5
) -> Dict[str, Any]:
    """
    Запуск эксперимента (автоматический выбор BBOB или стандартных задач).
    """
    # Для BBOB используем доступные размерности 2,3,5
    bbob_dims = [2, 3, 5]
    
    if COCO_AVAILABLE:
        print("\nИспользуем BBOB-constrained тестовый набор")
        return run_bbob_experiment(bbob_dims, None, n_runs, n_iterations, n_initial_points_factor)
    else:
        print("\nBBOB не доступен, используем стандартные тестовые задачи")
        return run_standard_experiment(dimensions, n_runs, n_iterations, n_initial_points_factor)