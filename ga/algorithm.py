import numpy as np
from typing import Callable, List
import multiprocessing as mp

def ga(
        initial_polulation: List,
        fitness_function: Callable,
        selection_callback: Callable,
        crossover_callback: Callable,
        mutation_callback: Callable,
        elite_fraction: float,
        number_of_generations: int
        ):
    pass