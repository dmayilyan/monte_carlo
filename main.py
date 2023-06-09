import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from tools import time_run


def get_shape() -> Tuple[np.ndarray, np.ndarray]:
    l = 0.03

    x = np.arange(0.001, 1.001, 0.001)
    y = l / x * np.exp(-l / x)

    return x, y


def get_ratio(y: np.ndarray) -> np.ndarray:
    return 1 / np.max(y)


def is_under(points: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    # :0 is x
    # :1 is y
    l = 0.03

    points[:, 0] = points[:, 0] + 0.001

    # Applying the function and scaling it
    y_to_test = l / points[:, 0] * np.exp(-l / points[:, 0])
    y_to_test *= ratio

    # checking for number being under the curve
    mask = points[:, 1] < y_to_test
    print(f"Filled by {np.count_nonzero(mask) / 1_000_000 * 100:.4}%")

    return points[:, 0][mask]


@time_run
def generate_curve():
    # TODO Generalize for any function and sample count
    x, y = get_shape()

    ratio = get_ratio(y)

    # Generating numbers and rounding for x index
    xy_dots = np.random.uniform(0, 0.999, size=(1_000_000, 2)).round(3)

    points_under = is_under(xy_dots, ratio)
    points_under *= 1000

    a, b = np.unique(points_under, return_counts=True)

    return np.unique(points_under, return_counts=True)


if __name__ == "__main__":
    # TODO Adapt to have easy access to other sampling methods
    x, y = generate_curve()

    plt.plot(x, y, "+")
    plt.show()
