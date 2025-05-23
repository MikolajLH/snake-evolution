import numpy as np

def uniform_crossover(parent1 : np.ndarray, parent2 : np.ndarray) -> np.ndarray:
    assert parent1.shape == parent2.shape

    offspring = parent1.copy()
    
    mask = np.random.uniform(0, 1, size= parent1.shape)

    offspring[mask > 0.5] = parent2[mask > 0.5]

    return offspring


def single_point_crossover(parent1 : np.ndarray, parent2 : np.ndarray) -> np.ndarray:
    assert parent1.shape == parent2.shape

    flat_p1 = parent1.flatten()
    flat_p2 = parent2.flatten()

    offspring = flat_p1.copy()

    break_point = np.random.randint(len(offspring))

    offspring[break_point:] = flat_p2[break_point:]

    return offspring.reshape(parent1.shape)


def double_point_crossover(parent1 : np.ndarray, parent2 : np.ndarray) -> np.ndarray:
    assert parent1.shape == parent2.shape
    
    flat_p1 = parent1.flatten()
    flat_p2 = parent2.flatten()

    offspring = flat_p1.copy()

    break_point_1 = np.random.randint(len(offspring))
    break_point_2 = np.random.randint(break_point_1, len(offspring))

    offspring[break_point_1:break_point_2] = flat_p2[break_point_1:break_point_2]
    
    return offspring.reshape(parent1.shape)