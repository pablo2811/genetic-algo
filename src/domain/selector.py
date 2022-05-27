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


class RouletteSelector(Selector):

    def __init__(self, population_size: int):
        self.population_size = population_size

    def select(self, population: Population) -> Population:

        total = sum([obs.evaluate() for obs in population.observations])

        probabilities = [0]
        for j in range(len(population.observations)):
            probabilities.append((population.observations[j].evaluate() / total) + probabilities[-1])

        selected = list()

        while len(selected) < self.population_size:

            roulette_spin = np.random.uniform()

            for j in range(len(population.observations)):
                if probabilities[j] < roulette_spin < probabilities[j + 1]:
                    selected.append(population.observations[j])

        return Population(selected)
