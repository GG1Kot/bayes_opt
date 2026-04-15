"""
Тестовые задачи оптимизации с ограничениями.

Содержит 5 классических задач для тестирования алгоритмов
байесовской оптимизации с ограничениями.

Автор: Elizaveta Surda
Дата: 2026
"""

import numpy as np
from typing import List, Dict, Any


def sphere_constrained(x: np.ndarray) -> float:
    """
    Сфера с ограничением.
    
    Целевая функция:
        f(x) = sum(x_i^2)
    
    Ограничение:
        sum(x_i) <= 1
    
    Глобальный минимум:
        f(x*) = 0 при x* = (0, 0, ..., 0)
    
    Параметры:
        x: вектор переменных
        
    Возвращает:
        значение функции (inf если ограничение нарушено)
    """
    if np.sum(x) > 1:
        return np.inf
    return np.sum(x**2)


def rosenbrock_constrained(x: np.ndarray) -> float:
    """
    Функция Розенброка с ограничением.
    
    Целевая функция:
        f(x) = sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    
    Ограничение:
        -2 <= x_i <= 2 для всех i
    
    Глобальный минимум:
        f(x*) = 0 при x* = (1, 1, ..., 1)
    
    Параметры:
        x: вектор переменных
        
    Возвращает:
        значение функции (inf если ограничение нарушено)
    """
    if np.any(x < -2) or np.any(x > 2):
        return np.inf
    
    n = len(x)
    f = 0
    for i in range(n - 1):
        f += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return f


def ackley_constrained(x: np.ndarray) -> float:
    """
    Функция Эккли с ограничением.
    
    Целевая функция:
        f(x) = -20*exp(-0.2*sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(2π*x_i))) + 20 + e
    
    Ограничение:
        sum(x_i^2) <= 10
    
    Глобальный минимум:
        f(x*) = 0 при x* = (0, 0, ..., 0)
    
    Параметры:
        x: вектор переменных
        
    Возвращает:
        значение функции (inf если ограничение нарушено)
    """
    if np.sum(x**2) > 10:
        return np.inf
    
    n = len(x)
    sum_sq = np.sum(x**2) / n
    sum_cos = np.sum(np.cos(2 * np.pi * x)) / n
    
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq)) - np.exp(sum_cos) + 20 + np.e


def rastrigin_constrained(x: np.ndarray) -> float:
    """
    Функция Растригина с ограничением.
    
    Целевая функция:
        f(x) = 10*n + sum(x_i^2 - 10*cos(2π*x_i))
    
    Ограничение:
        sum(x_i) <= 5
    
    Глобальный минимум:
        f(x*) = 0 при x* = (0, 0, ..., 0)
    
    Параметры:
        x: вектор переменных
        
    Возвращает:
        значение функции (inf если ограничение нарушено)
    """
    if np.sum(x) > 5:
        return np.inf
    
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def michalewicz_constrained(x: np.ndarray) -> float:
    """
    Функция Михалевича с ограничением.
    
    Целевая функция:
        f(x) = -sum_{i=1}^{n} sin(x_i) * sin(i * x_i^2 / π)^20
    
    Ограничение:
        0 <= x_i <= π для всех i
    
    Глобальный минимум зависит от размерности.
    Для n=2: f(x*) ≈ -1.8013 при x* ≈ (2.20, 1.57)
    
    Параметры:
        x: вектор переменных
        
    Возвращает:
        значение функции (inf если ограничение нарушено)
    """
    if np.any(x < 0) or np.any(x > np.pi):
        return np.inf
    
    n = len(x)
    f = 0
    for i in range(n):
        f += np.sin(x[i]) * (np.sin((i + 1) * x[i]**2 / np.pi))**20
    
    return -f


def get_problem_bounds(name: str, dimension: int) -> np.ndarray:
    """
    Получение границ для тестовой задачи.
    
    Параметры:
        name: название задачи
        dimension: размерность
        
    Возвращает:
        bounds: массив границ формы (dimension, 2)
    """
    bounds_dict = {
        'sphere': np.array([[-5, 5]] * dimension),
        'rosenbrock': np.array([[-2, 2]] * dimension),
        'ackley': np.array([[-5, 5]] * dimension),
        'rastrigin': np.array([[-5, 5]] * dimension),
        'michalewicz': np.array([[0, np.pi]] * dimension),
    }
    return bounds_dict.get(name, np.array([[-5, 5]] * dimension))


def get_test_problems(dimensions: List[int]) -> List[Dict[str, Any]]:
    """
    Получение списка тестовых задач для эксперимента.
    
    Параметры:
        dimensions: список размерностей
        
    Возвращает:
        problems: список словарей с информацией о задачах
    """
    problems = {
        'Sphere': sphere_constrained,
        'Rosenbrock': rosenbrock_constrained,
        'Ackley': ackley_constrained,
        'Rastrigin': rastrigin_constrained,
        'Michalewicz': michalewicz_constrained,
    }
    
    result = []
    for name, func in problems.items():
        for dim in dimensions:
            bounds = get_problem_bounds(name.lower(), dim)
            result.append({
                'name': name,
                'function': func,
                'dimension': dim,
                'bounds': bounds
            })
    
    return result