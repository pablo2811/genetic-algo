import random
from abc import ABC, abstractmethod
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


class TournamentSelector(Selector):

    def __init__(self, tournament_size: int, population_size: int, elitism_thresh: int = 0):
        self.elitism_thresh = elitism_thresh
        self.population_size = population_size
        self.tournament_size = tournament_size

    def select(self, population: Population) -> Population:
        selected = population.n_best_observations(self.elitism_thresh)

        while len(selected) < self.population_size:
            tournament = random.sample(population.observations, self.tournament_size)
            sorted(tournament, key=lambda obs: obs.evaluate())
            selected.append(tournament[-1])

        return Population(selected)



