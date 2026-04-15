"""
Тестовые задачи оптимизации с ограничениями.
"""

import numpy as np
from typing import Tuple


def sphere_constrained(x: np.ndarray) -> float:
    """
    Сфера с ограничением.
    
    Минимизировать: f(x) = sum(x_i^2)
    Ограничение: sum(x_i) <= 1
    
    Глобальный минимум: f(x*) = 0 при x* = (0,...,0)
    """
    n = len(x)
    
    # Ограничение
    if np.sum(x) > 1:
        return np.inf
    
    # Целевая функция
    return np.sum(x**2)


def rosenbrock_constrained(x: np.ndarray) -> float:
    """
    Функция Розенброка с ограничением.
    
    Минимизировать: f(x) = sum(100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2)
    Ограничение: x[i] >= -2, x[i] <= 2
    
    Глобальный минимум: f(x*) = 0 при x* = (1,...,1)
    """
    n = len(x)
    
    # Ограничения
    if np.any(x < -2) or np.any(x > 2):
        return np.inf
    
    # Целевая функция
    f = 0
    for i in range(n - 1):
        f += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    
    return f


def ackley_constrained(x: np.ndarray) -> float:
    """
    Функция Эккли с ограничением.
    
    Минимизировать: f(x) = -20*exp(-0.2*sqrt(1/n*sum(x_i^2))) - exp(1/n*sum(cos(2π*x_i))) + 20 + e
    Ограничение: sum(x_i^2) <= 10
    
    Глобальный минимум: f(x*) = 0 при x* = (0,...,0)
    """
    n = len(x)
    
    # Ограничение
    if np.sum(x**2) > 10:
        return np.inf
    
    # Целевая функция
    sum_sq = np.sum(x**2) / n
    sum_cos = np.sum(np.cos(2 * np.pi * x)) / n
    
    f = -20 * np.exp(-0.2 * np.sqrt(sum_sq)) - np.exp(sum_cos) + 20 + np.e
    
    return f


def rastrigin_constrained(x: np.ndarray) -> float:
    """
    Функция Растригина с ограничением.
    
    Минимизировать: f(x) = 10*n + sum(x_i^2 - 10*cos(2π*x_i))
    Ограничение: sum(x_i) <= 5
    
    Глобальный минимум: f(x*) = 0 при x* = (0,...,0)
    """
    n = len(x)
    
    # Ограничение
    if np.sum(x) > 5:
        return np.inf
    
    # Целевая функция
    f = 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    return f


def michalewicz_constrained(x: np.ndarray) -> float:
    """
    Функция Михалевича с ограничением.
    
    Минимизировать: f(x) = -sum(sin(x_i) * sin(i*x_i^2/π)^20)
    Ограничение: 0 <= x_i <= π
    
    Глобальный минимум зависит от размерности
    """
    n = len(x)
    
    # Ограничения
    if np.any(x < 0) or np.any(x > np.pi):
        return np.inf
    
    # Целевая функция
    f = 0
    for i in range(n):
        f += np.sin(x[i]) * (np.sin((i + 1) * x[i]**2 / np.pi))**20
    
    return -f


# Функции G01-G05 оставляем для справки, но они не используются в эксперименте
def g01_constrained(x: np.ndarray) -> float:
    """Задача G01: 13 переменных."""
    n = len(x)
    if n != 13:
        return np.inf
    
    x1_4 = x[:4]
    f = 5 * np.sum(x1_4) - 5 * np.sum(x1_4**2) - np.sum(x[4:])
    
    constraints = []
    constraints.append(2*x[0] + 2*x[1] + x[9] + x[10] - 10)
    constraints.append(2*x[0] + 2*x[2] + x[9] + x[11] - 10)
    constraints.append(2*x[1] + 2*x[2] + x[10] + x[11] - 10)
    constraints.append(-8*x[0] + x[9])
    constraints.append(-8*x[1] + x[10])
    constraints.append(-8*x[2] + x[11])
    constraints.append(-2*x[3] - x[4] + x[9])
    constraints.append(-2*x[5] - x[6] + x[10])
    constraints.append(-2*x[7] - x[8] + x[11])
    
    for constraint in constraints:
        if constraint > 1e-6:
            return np.inf
    
    return f


def g02_constrained(x: np.ndarray) -> float:
    """Задача G02: 20 переменных."""
    n = len(x)
    if n != 20:
        return np.inf
    
    prod_x = np.prod(x)
    sum_x = np.sum(x)
    
    if prod_x <= 0.75:
        return np.inf
    if sum_x >= 7.5 * n:
        return np.inf
    
    cos_x = np.cos(x)
    numerator = abs(np.sum(cos_x**4) - 2 * np.prod(cos_x**2))
    denominator = np.sqrt(np.sum(np.arange(1, n+1) * x**2))
    
    return -numerator / denominator


def g03_constrained(x: np.ndarray) -> float:
    """Задача G03: 10 переменных."""
    n = 10
    if len(x) != n:
        return np.inf
    
    sum_squares = np.sum(x**2)
    if abs(sum_squares - 1) > 1e-6:
        return np.inf
    if np.any(x <= 0):
        return np.inf
    
    return -(np.sqrt(n))**n * np.prod(x)


def g04_constrained(x: np.ndarray) -> float:
    """Задача G04: 5 переменных."""
    if len(x) != 5:
        return np.inf
    
    x1, x2, x3, x4, x5 = x
    
    f = 5.3578547 * x3**2 + 0.8356891 * x1 * x5 + 37.293239 * x1 - 40792.141
    
    g1 = 85.334407 + 0.0056858 * x2 * x5 + 0.0006262 * x1 * x4 - 0.0022053 * x3 * x5 - 92
    g2 = -85.334407 - 0.0056858 * x2 * x5 - 0.0006262 * x1 * x4 + 0.0022053 * x3 * x5
    g3 = 80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3**2 - 110
    g4 = -80.51249 - 0.0071317 * x2 * x5 - 0.0029955 * x1 * x2 - 0.0021813 * x3**2 + 90
    g5 = 9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4 - 25
    g6 = -9.300961 - 0.0047026 * x3 * x5 - 0.0012547 * x1 * x3 - 0.0019085 * x3 * x4 + 20
    
    constraints = [g1, g2, g3, g4, g5, g6]
    for constraint in constraints:
        if constraint > 1e-6:
            return np.inf
    
    return f


def g05_constrained(x: np.ndarray) -> float:
    """Задача G05: 4 переменные."""
    if len(x) != 4:
        return np.inf
    
    x1, x2, x3, x4 = x
    
    f = 3*x1 + 0.000001*x1**3 + 2*x2 + (0.000002/3)*x2**3
    
    g1 = -x4 + x3 - 0.55
    g2 = -x3 + x4 - 0.55
    h1 = 1000*np.sin(-x3 - 0.25) + 1000*np.sin(-x4 - 0.25) + 894.8 - x1
    h2 = 1000*np.sin(x3 - 0.25) + 1000*np.sin(x3 - x4 - 0.25) + 894.8 - x2
    h3 = 1000*np.sin(x4 - 0.25) + 1000*np.sin(x4 - x3 - 0.25) + 1294.8
    
    if abs(h1) > 1e-4 or abs(h2) > 1e-4 or abs(h3) > 1e-4:
        return np.inf
    if g1 > 1e-6 or g2 > 1e-6:
        return np.inf
    
    return f


def get_problem_bounds(name: str, dimension: int) -> np.ndarray:
    """Получение границ для тестовой задачи."""
    bounds_dict = {
        'sphere': np.array([[-5, 5]] * dimension),
        'rosenbrock': np.array([[-2, 2]] * dimension),
        'ackley': np.array([[-5, 5]] * dimension),
        'rastrigin': np.array([[-5, 5]] * dimension),
        'michalewicz': np.array([[0, np.pi]] * dimension),
        'g01': np.array([[0, 1]] * 13),
        'g02': np.array([[0, 10]] * 20),
        'g03': np.array([[0, 1]] * 10),
        'g04': np.array([[78, 102], [33, 45], [27, 45], [27, 45], [27, 45]]),
        'g05': np.array([[0, 1200], [0, 1200], [-0.55, 0.55], [-0.55, 0.55]]),
    }
    return bounds_dict.get(name, np.array([[-5, 5]] * dimension))


def get_test_problems(dimensions: list):
    """Получение списка тестовых задач с переменной размерностью."""
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