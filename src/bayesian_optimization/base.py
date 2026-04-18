"""
Базовый класс байесовского оптимизатора.

Реализует основной цикл оптимизации с использованием Gaussian Process
в качестве суррогатной модели и Expected Improvement как acquisition функции.

Автор: Elizaveta Surda
Дата: 2026
"""

import abc
import warnings
from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from src.experimental_design import lhs


class ConstraintHandler(abc.ABC):
    """
    Абстрактный базовый класс для методов учёта ограничений.

    Определяет интерфейс для PenaltyMethod, BarrierMethod,
    LagrangeMethod и ConstrainedExpectedImprovement.
    """

    @abc.abstractmethod
    def evaluate_constraints(self, X: np.ndarray) -> np.ndarray:
        """
        Суммарное нарушение ограничений.

        Аргументы:
            X: точки, форма (n_points, n_dims)

        Возвращает:
            violations: нарушения, форма (n_points,); 0 — допустима
        """

    @abc.abstractmethod
    def compute_penalized_objective(
        self, X: np.ndarray, f_values: np.ndarray
    ) -> np.ndarray:
        """
        Штрафованная целевая функция.

        Аргументы:
            X: точки, форма (n_points, n_dims)
            f_values: значения целевой функции, форма (n_points,)

        Возвращает:
            penalized: штрафованные значения, форма (n_points,)
        """

    @abc.abstractmethod
    def get_acquisition_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Веса для acquisition функции.

        Аргументы:
            X: точки, форма (n_points, n_dims)

        Возвращает:
            weights: веса в [0, 1], форма (n_points,)
        """

    @abc.abstractmethod
    def is_feasible(self, X: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        """
        Проверка допустимости точек.

        Аргументы:
            X: точки, форма (n_points, n_dims)
            tolerance: допуск на нарушение ограничений

        Возвращает:
            feasible: булев массив
        """


class BayesianOptimizer:
    """
    Байесовский оптимизатор с поддержкой ограничений.

    Алгоритм:
        1. LHS начальная выборка
        2. Обучение GP
        3. Максимизация EI → следующая точка
        4. Оценка целевой функции
        5. Повторение шагов 2–4

    Атрибуты:
        objective: целевая функция f(x) -> float
        bounds: границы переменных, форма (n_dims, 2)
        constraint_handler: метод учёта ограничений
        n_initial_points: размер начальной LHS-выборки
        random_state: seed для воспроизводимости
    """

    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        constraint_handler: ConstraintHandler | None = None,
        n_initial_points: int = 10,
        random_state: int = 42,
    ) -> None:
        """
        Инициализация оптимизатора.

        Аргументы:
            objective_function: минимизируемая функция
            bounds: границы переменных, форма (n_dims, 2)
            constraint_handler: метод учёта ограничений
            n_initial_points: размер начальной LHS-выборки
            random_state: seed для воспроизводимости
        """
        self.objective = objective_function
        self.bounds = np.array(bounds, dtype=float)
        self.n_dims = self.bounds.shape[0]
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        self.constraint_handler = constraint_handler

        kernel = Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=random_state,
        )

        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.penalized_y: np.ndarray | None = None

        np.random.seed(random_state)

    def _initial_sample(self) -> np.ndarray:
        """
        Начальная LHS-выборка.

        Возвращает:
            X_init: точки, форма (n_initial_points, n_dims)
        """
        return lhs.latin_hypercube_sample(
            self.bounds, self.n_initial_points, self.random_state
        )

    def _update_model(self) -> None:
        """Переобучение GP на всех конечных накопленных данных."""
        if self.X is not None and len(self.X) > self.n_initial_points:
            finite_mask = np.isfinite(self.penalized_y)
            if np.sum(finite_mask) > self.n_initial_points:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.gp.fit(self.X[finite_mask], self.penalized_y[finite_mask])

    def _acquisition_function(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """
        Expected Improvement (EI).

        EI(x) = E[max(f_best - f(x), 0)]

        Аргументы:
            X: точки, форма (n_points, n_dims)
            xi: параметр баланса exploration/exploitation

        Возвращает:
            ei: значения EI, форма (n_points,)
        """
        X = np.atleast_2d(X)
        finite_mask = np.isfinite(self.penalized_y)

        if np.sum(finite_mask) > 0:
            mu, sigma = self.gp.predict(X, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            f_best = float(np.min(self.penalized_y[finite_mask]))
        else:
            mu = np.zeros(len(X))
            sigma = np.ones(len(X))
            f_best = 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            z = (f_best - mu - xi) / sigma
            ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
            ei[sigma == 0] = 0.0
            ei[ei < 0] = 0.0

        if self.constraint_handler is not None:
            ei *= self.constraint_handler.get_acquisition_weights(X)

        return ei

    def _next_point(self, n_restarts: int = 5) -> np.ndarray:
        """
        Следующая точка через максимизацию EI (L-BFGS-B с рестартами).

        Аргументы:
            n_restarts: количество случайных стартов оптимизатора

        Возвращает:
            x_next: следующая точка для оценки
        """
        best_x: np.ndarray | None = None
        best_acq = -np.inf
        fallback = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

        for _ in range(n_restarts):
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            try:
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1)),
                    x0,
                    bounds=self.bounds,
                    method="L-BFGS-B",
                    options={"maxiter": 100},
                )
                if result.success and float(-result.fun) > best_acq:
                    best_acq = float(-result.fun)
                    best_x = result.x
            except Exception:  # pylint: disable=broad-except
                continue

        return best_x if best_x is not None else fallback

    def optimize(
        self, n_iterations: int, verbose: bool = False
    ) -> tuple[float, np.ndarray, list[float]]:
        """
        Основной цикл оптимизации.

        Аргументы:
            n_iterations: число итераций
            verbose: выводить прогресс в консоль

        Возвращает:
            best_value: лучшее найденное значение целевой функции
            best_point: лучшая найденная точка
            history: история лучших значений по итерациям
        """
        self.X = self._initial_sample()
        self.y = np.array([self.objective(x) for x in self.X])

        if self.constraint_handler is not None:
            self.penalized_y = self.constraint_handler.compute_penalized_objective(
                self.X, self.y
            )
            feasible_mask = self.constraint_handler.is_feasible(self.X)
        else:
            self.penalized_y = self.y.copy()
            feasible_mask = np.ones(len(self.X), dtype=bool)

        if np.any(feasible_mask):
            best_idx = int(np.argmin(self.y[feasible_mask]))
            best_value = float(self.y[feasible_mask][best_idx])
            best_point = self.X[feasible_mask][best_idx].copy()
        else:
            best_idx = int(np.argmin(self.penalized_y))
            best_value = float(self.y[best_idx])
            best_point = self.X[best_idx].copy()

        history: list[float] = [best_value]

        for iteration in range(n_iterations):
            self._update_model()
            next_x = self._next_point()
            next_y = float(self.objective(next_x))

            self.X = np.vstack([self.X, next_x.reshape(1, -1)])
            self.y = np.append(self.y, next_y)

            if self.constraint_handler is not None:
                penalized = self.constraint_handler.compute_penalized_objective(
                    next_x.reshape(1, -1), np.array([next_y])
                )
                self.penalized_y = np.append(self.penalized_y, penalized[0])
                feasible_mask = self.constraint_handler.is_feasible(self.X)
            else:
                self.penalized_y = np.append(self.penalized_y, next_y)
                feasible_mask = np.ones(len(self.X), dtype=bool)

            if np.any(feasible_mask):
                current_best = float(np.min(self.y[feasible_mask]))
                if current_best < best_value:
                    best_value = current_best
                    best_point = self.X[feasible_mask][
                        int(np.argmin(self.y[feasible_mask]))
                    ].copy()

            history.append(best_value)

            if verbose:
                print(f"  Итерация {iteration + 1}/{n_iterations}: {best_value:.6f}")

        return best_value, best_point, history