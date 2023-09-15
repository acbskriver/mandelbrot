import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers.legacy import Adam


def mandelbrot_dataset_creator(
    real_start: float = -2.25,
    real_end: float = 0.75,
    real_nr_pixels: int = 1000,
    imag_start: complex = -1j,
    imag_end: complex = 1j,
    imag_nr_pixels: int = 1000,
    initial_z: complex = 0,
    exponent: complex = 2,
    nr_iterations: int = 100,
    classification_threshold: float = 1,
) -> pd.DataFrame:
    """
    Performs mandelbrot iteration with params:
    --> initial_z, exponent, nr_iterations

    on complex grid with grid params:
    --> real_start, real_end, imag_start, imag_end, real_nr_pixels, imag_nr_pixels

    returns df of resulting number of iterations, the grid points needed to cross the:
    --> classification_threshold
    """

    # define the Mandelbrot function:
    mandelbrot_funct = lambda z, c, exp: z**exp + c

    # create c as as complex grid of specified size
    c_real = np.linspace(real_start, real_end, real_nr_pixels)
    c_imag = np.linspace(imag_start, imag_end, imag_nr_pixels)
    c_real_grid, c_imag_grid = np.meshgrid(c_real, c_imag)
    c = c_real_grid + c_imag_grid

    # init z grid  with initial z & dataset grid with initial value 0
    z = np.full_like(c, initial_z, dtype=complex)
    dataset = np.full_like(c, 0, dtype=int)

    # perform the mandelbrot iteration
    # dataset stores number of iterations it took for grid point to diverge
    for i in range(nr_iterations):
        z = mandelbrot_funct(z, c, exponent)
        not_diverged_grid_points = np.abs(z) <= classification_threshold
        dataset[not_diverged_grid_points] = i

        # handle overflow, by resetting z to two
        diverged_grid_points = np.abs(z) > classification_threshold * 10
        z[diverged_grid_points] = 10
        c[diverged_grid_points] = 10

    # convert to df, with: col_name = a; row_name = b as in c = a + bi for each original grid pt
    dataset = pd.DataFrame(dataset).rename(
        columns={dflt: c_real for dflt, c_real in enumerate(c_real)},
        index={dflt: c_imag.imag for dflt, c_imag in enumerate(c_imag)},
    )

    # return dataset
    return dataset


def plot_mandelbrot_dataset(
    dataset: pd.DataFrame,
    fig_size: Tuple = (15, 15),
    title: str = None,
    show_color_bar: bool = False,
) -> None:
    """plots mandelbrot fractal from dataset"""

    # plot mandelbrot image
    plt.figure(figsize=fig_size)
    plt.imshow(
        dataset,
        cmap="plasma",
        aspect="equal",
        extent=[
            float(dataset.columns[0]),
            float(dataset.columns[-1]),
            float(dataset.index[0]),
            float(dataset.index[-1]),
        ],
    )
    plt.title(title)
    if show_color_bar:
        plt.colorbar()
    plt.show

    return None


def initialize_model():
    """Create model architecture"""

    model = Sequential()

    model.add(Dense(32, activation=LeakyReLU(), input_shape=(2,)))
    model.add(Dense(32, activation=LeakyReLU(alpha=0.01)))
    model.add(Dense(32, activation=LeakyReLU(alpha=0.01)))
    model.add(Dense(32, activation=LeakyReLU(alpha=0.01)))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mae"])

    return model
