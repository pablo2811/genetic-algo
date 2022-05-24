import random
from copy import copy

import numpy as np

from src.domain.population import Population, Observation, PopulationDevelopment
from src.domain.selector import Selector
from src.domain.stop_condition import StopCondition


class CrossOver:

    def __init__(self, probability: float):
        self.probability = probability

    def execute(self, population: Population) -> Population:

        observations = copy(population.observations)
        random.shuffle(observations)

        for i in range(0, population.size() - 1, 2):

            if np.random.uniform() < self.probability:
                first, second = observations[i], observations[i + 1]
                observations.append(first.crossover(second))

        return Population(observations)


class Mutator:

    def __init__(self, probability: float):
        self.probability = probability

    def execute(self, population: Population) -> Population:
        observations = population.observations
        added = list()

        for observation in population.observations:
            if np.random.uniform() < self.probability:
                added.append(observation.mutate())

        return Population(observations + added)


class GeneticAlgorithm:
    def __init__(self,
                 stop_condition: StopCondition,
                 select: Selector,
                 crossover_probability: float,
                 mutation_probability: float,
                 ):
        self.select = select
        self.mutate = Mutator(mutation_probability)
        self.cross_over = CrossOver(crossover_probability)
        self.stop_condition = stop_condition

    def run(self, population: Population) -> Observation:
        population_development = PopulationDevelopment.defaulter()

        while not self.stop_condition.stop(population_development):
            # perform crossover
            population = self.cross_over.execute(population)

            # perform mutation
            population = self.mutate.execute(population)

            # perform selection
            population = self.select.select(population)

            # update population development
            population_development.update(population)

        return population.best_observation()
