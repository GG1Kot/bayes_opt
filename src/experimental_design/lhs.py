"""Модуль с методами планирования эксперимента."""

import numpy as np


def latin_hypercube_sample(bounds, n_samples, random_state=None):
    """
    Генерация выборки методом латинского гиперкуба.
    
    Аргументы:
        bounds: границы переменных, форма (n_dims, 2)
        n_samples: количество точек
        random_state: seed для воспроизводимости
    
    Returns:
        массив точек формы (n_samples, n_dims)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_dims = bounds.shape[0]
    samples = np.zeros((n_samples, n_dims))
    
    for i in range(n_dims):
        seg_size = (bounds[i, 1] - bounds[i, 0]) / n_samples
        points = bounds[i, 0] + seg_size * (np.random.random(n_samples) + np.arange(n_samples))
        np.random.shuffle(points)
        samples[:, i] = points
    
    return samples


def random_sample(bounds, n_samples, random_state=None):
    """
    Генерация случайной выборки.
    
    Аргументы:
        bounds: границы переменных, форма (n_dims, 2)
        n_samples: количество точек
        random_state: seed для воспроизводимости
    
    Returns:
        массив точек формы (n_samples, n_dims)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_dims = bounds.shape[0]
    samples = np.zeros((n_samples, n_dims))
    
    for i in range(n_dims):
        samples[:, i] = np.random.uniform(bounds[i, 0], bounds[i, 1], n_samples)
    
    return samples


def lhs_initialize(bounds, n_samples, random_state=None):
    """
    Инициализация выборки методом LHS (синоним latin_hypercube_sample).
    
    Аргументы:
        bounds: границы переменных, форма (n_dims, 2)
        n_samples: количество точек
        random_state: seed для воспроизводимости
    
    Returns:
        массив точек формы (n_samples, n_dims)
    """
    return latin_hypercube_sample(bounds, n_samples, random_state)