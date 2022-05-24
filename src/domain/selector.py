from abc import ABC, abstractmethod
from copy import copy

import numpy as np

from src.domain.population import Population


class Selector(ABC):
    @abstractmethod
    def select(self, population: Population) -> Population:
        pass


class SimpleSelector(Selector):

    def __init__(self, population_size: int):
        self.population_size = population_size

    def select(self, population: Population) -> Population:
        return Population(population.n_best_observations(self.population_size))


# TODO: fix RouletteSelector
class RouletteSelector(Selector):

    def __init__(self, population_size: int):
        self.population_size = population_size

    def select(self, population: Population) -> Population:

        observations = copy(population.observations)
        selected = list()

        for i in range(self.population_size):
            total = sum([obs.evaluate() for obs in observations])
            probs = [obs.evaluate() / total for obs in observations]

            roulette_spin = np.random.uniform()
            arrow = 0.0
            i = 0
            while arrow < roulette_spin:
                arrow += probs[i]
                i += 1

            selected.append(observations[i - 1])
            observations.pop(i - 1)  # slow

        return Population(selected)
