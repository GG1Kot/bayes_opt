"""
Тестовые задачи оптимизации с ограничениями.

Каждая задача представлена отдельной целевой функцией и набором
функций ограничений g_i(x) <= 0.

Автор: Elizaveta Surda
Дата: 2026
"""

from collections.abc import Callable
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Целевые функции
# ---------------------------------------------------------------------------


def sphere_objective(x: np.ndarray) -> float:
    """
    Функция Сферы: f(x) = sum(x_i^2).

    Глобальный минимум: f(0,...,0) = 0.

    Аргументы:
        x: вектор переменных

    Возвращает:
        значение функции
    """
    return float(np.sum(x**2))


def rosenbrock_objective(x: np.ndarray) -> float:
    """
    Функция Розенброка: f(x) = sum[100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2].

    Глобальный минимум: f(1,...,1) = 0.

    Аргументы:
        x: вектор переменных

    Возвращает:
        значение функции
    """
    n = len(x)
    return float(sum(
        100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        for i in range(n - 1)
    ))


def ackley_objective(x: np.ndarray) -> float:
    """
    Функция Эккли.

    f(x) = -20*exp(-0.2*sqrt(sum(x^2)/n)) - exp(sum(cos(2pi*x))/n) + 20 + e

    Глобальный минимум: f(0,...,0) = 0.

    Аргументы:
        x: вектор переменных

    Возвращает:
        значение функции
    """
    n = len(x)
    sum_sq = np.sum(x**2) / n
    sum_cos = np.sum(np.cos(2 * np.pi * x)) / n
    return float(-20 * np.exp(-0.2 * np.sqrt(sum_sq)) - np.exp(sum_cos) + 20 + np.e)


def rastrigin_objective(x: np.ndarray) -> float:
    """
    Функция Растригина: f(x) = 10n + sum(x_i^2 - 10*cos(2pi*x_i)).

    Глобальный минимум: f(0,...,0) = 0.

    Аргументы:
        x: вектор переменных

    Возвращает:
        значение функции
    """
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def michalewicz_objective(x: np.ndarray) -> float:
    """
    Функция Михалевича: f(x) = -sum(sin(x_i) * sin(i*x_i^2/pi)^20).

    Глобальный минимум зависит от размерности.

    Аргументы:
        x: вектор переменных

    Возвращает:
        значение функции
    """
    n = len(x)
    return float(-sum(
        np.sin(x[i]) * (np.sin((i + 1) * x[i] ** 2 / np.pi)) ** 20
        for i in range(n)
    ))

def miura_ori_geometry(a: float, b: float, alpha_deg: float) -> tuple[float, float, float]:
    """
    Вычисляет геометрические характеристики ячейки Миура-Ори.
    
    Аргументы:
        a: длина стороны a (мм)
        b: длина стороны b (мм)
        alpha_deg: секторный угол (градусы)
    
    Возвращает:
        folded_volume: сложенный объём (отн. ед.)
        stiffness: жёсткость (отн. ед.)
        poisson_ratio: коэффициент Пуассона
    """
    alpha_rad = np.radians(alpha_deg)
    
    # Сложенный объём (минимизируем)
    folded_volume = np.cos(alpha_rad)
    
    # Жёсткость (максимизируем)
    stiffness = np.sin(2 * alpha_rad) * (a + b) / (a * b + 1e-6)
    
    # Коэффициент Пуассона (должен быть отрицательным)
    poisson_ratio = -np.tan(alpha_rad) ** 2
    
    return folded_volume, stiffness, poisson_ratio

def miura(x: np.ndarray) -> float:
    """
    Целевая функция для оптимизации геометрии Миура-Ори.
    
    Минимизируем сложенный объём.
    
    Аргументы:
        x: вектор переменных [a, b, alpha]
    
    Возвращает:
        значение целевой функции (сложенный объём)
    """
    a, b, alpha = x[0], x[1], x[2]
    
    # Физическая проверка (минимальные границы)
    if a <= 0 or b <= 0 or alpha <= 0 or alpha >= 90:
        return 1e10  # Огромный штраф за невалидные параметры
    
    folded_volume, stiffness, _ = miura_ori_geometry(a, b, alpha)
    
    return folded_volume/stiffness

# ---------------------------------------------------------------------------
# Функции ограничений: g(x) <= 0 означает допустимость
# ---------------------------------------------------------------------------


def _sphere_constraint(x: np.ndarray) -> float:
    """Ограничение Сферы: sum(x) <= 1."""
    return float(np.sum(x) - 1.0)


def _rosenbrock_constraint(x: np.ndarray) -> float:
    """Ограничение Розенброка: sum(x^2) <= n (сфера радиуса sqrt(n))."""
    return float(np.sum(x**2) - float(len(x)))


def _ackley_constraint(x: np.ndarray) -> float:
    """Ограничение Эккли: sum(x^2) <= 5."""
    return float(np.sum(x**2) - 5.0)


def _rastrigin_constraint(x: np.ndarray) -> float:
    """Ограничение Растригина: sum(x) <= 1."""
    return float(np.sum(x) - 1.0)


def _michalewicz_constraint(x: np.ndarray) -> float:
    """Ограничение Михалевича: sum(x) <= 0.6*n*pi (активное на 40% области)."""
    return float(np.sum(x) - 0.6 * len(x) * np.pi)

def _miura_constraint_stiffness(x: np.ndarray) -> float:
    """
    Ограничение для геометрии Миура-Ори.
    
    Возвращает значение <= 0 для допустимых точек.
    
    Аргументы:
        x: вектор переменных [a, b, alpha]
    
    Возвращает:
        значение ограничения (должно быть <= 0 для допустимости)
    """
    a, b, alpha = x[0], x[1], x[2]
    
    # Вычисляем характеристики
    _, stiffness,_ = miura_ori_geometry(a, b, alpha)
    
    # Ограничение 1: Жёсткость должна быть >= 0.5
    # Возвращаем 0.5 - stiffness (должно быть <= 0)
    return float(0.5 - stiffness)
def _miura_constraint_stiffness_poisson_ratio(x: np.ndarray) -> float:
    a, b, alpha = x[0], x[1], x[2]
    
    # Вычисляем характеристики
    _, _,poisson_ratio = miura_ori_geometry(a, b, alpha)
    # Ограничение 2: Коэффициент Пуассона должен быть < 0
    # Возвращаем poisson_ratio (должно быть <= 0)
    return poisson_ratio  # poisson_ratio уже отрицательный в норме
def _miura_constraint_stiffness_aspect_ratio(x: np.ndarray) -> float:
    a, b = x[0], x[1]
    # Ограничение 3: Пропорции (a/b не слишком экстремальные)
    aspect_ratio = a / b
    return max(0.1 - aspect_ratio, aspect_ratio - 10.0)  # 0.1 <= a/b <= 10

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def get_problem_bounds(name: str, dimension: int) -> np.ndarray:
    """
    Границы переменных для тестовой задачи.

    Аргументы:
        name: название задачи (нижний регистр)
        dimension: размерность

    Возвращает:
        bounds: массив формы (dimension, 2)
    """
    bounds_map: dict[str, np.ndarray] = {
        "sphere":      np.array([[-5.0, 5.0]] * dimension),
        "rosenbrock":  np.array([[-2.0, 2.0]] * dimension),
        "ackley":      np.array([[-5.0, 5.0]] * dimension),
        "rastrigin":   np.array([[-5.0, 5.0]] * dimension),
        "michalewicz": np.array([[0.0, np.pi]] * dimension),
        "miura":       np.array([[1.0, 10.0],[1.0, 10.0],[0.0, 90.0]]),
    }
    return bounds_map.get(name.lower(), np.array([[-5.0, 5.0]] * dimension))


def get_test_problems(dimensions: list[int]) -> list[dict[str, Any]]:
    """
    Список тестовых задач для эксперимента.

    Каждый элемент содержит:
        name: название задачи
        function: целевая функция f(x) -> float
        constraints: список функций g_i(x) <= 0
        dimension: размерность
        bounds: границы переменных, форма (dimension, 2)

    Аргументы:
        dimensions: список размерностей

    Возвращает:
        список словарей с описанием задач
    """
    problems_config: list[tuple[str, Callable, list[Callable]]] = [
        ("Sphere",      sphere_objective,     [_sphere_constraint]),
        ("Rosenbrock",  rosenbrock_objective, [_rosenbrock_constraint]),
        ("Ackley",      ackley_objective,     [_ackley_constraint]),
        ("Rastrigin",   rastrigin_objective,  [_rastrigin_constraint]),
        ("Michalewicz", michalewicz_objective, [_michalewicz_constraint]),
        ("miura",       miura, [_miura_constraint_stiffness, _miura_constraint_stiffness_poisson_ratio, _miura_constraint_stiffness_aspect_ratio]),
    ]

    result: list[dict[str, Any]] = []
    for name, objective, constraints in problems_config:
        if name=="miura":
            bounds = get_problem_bounds(name.lower(), 3)
            result.append({
                "name": name,
                "function": objective,
                "constraints": constraints,
                "dimension": 3,
                "bounds": bounds,
            })
        else:
            for dim in dimensions:
                bounds = get_problem_bounds(name.lower(), dim)
                result.append({
                    "name": name,
                    "function": objective,
                    "constraints": constraints,
                    "dimension": dim,
                    "bounds": bounds,
                })

    return result