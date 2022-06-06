import random
from copy import copy
from dataclasses import dataclass

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

    def execute(self, population: Population) -> tuple[Population, float]:
        observations = population.observations
        added = list()
        improvement_mutation = 0

        for observation in population.observations:
            if np.random.uniform() < self.probability:
                mutated = observation.mutate()

                if mutated.evaluate() > observation.evaluate():
                    improvement_mutation += 1

                added.append(observation.mutate())

        improved_mutation_factor = self.get_imf(improvement_mutation, len(added))

        return Population(observations + added), improved_mutation_factor

    @staticmethod
    def get_imf(improvement_mutation, added_count) -> float:
        if added_count == 0:
            return 0

        return improvement_mutation / added_count


@dataclass
class GeneticAlgorithmConfiguration:
    enable_mutation_proba_evolution: bool = False
    mutation_evolution_step: int = 30
    mutation_proba_delta: float = 0.005
    crossover_proba: float = 0.7
    mutation_proba: float = 0.3
    verbose_step: int = 30

    def update(self, population_development: PopulationDevelopment):
        if self.enable_mutation_proba_evolution and population_development.n_iter > self.mutation_evolution_step:
            last_iterations_improvements = population_development.improved_mutations[-self.mutation_evolution_step:]

            if all(last_iterations_improvements):
                self.mutation_proba = min(self.mutation_proba + self.mutation_proba_delta, 1)
            else:
                self.mutation_proba = max(self.mutation_proba - self.mutation_proba_delta, 0)


class GeneticAlgorithm:
    def __init__(self,
                 stop_condition: StopCondition,
                 select: Selector,
                 genetic_algo_config: GeneticAlgorithmConfiguration
                 ):
        self.select = select
        self.stop_condition = stop_condition
        self.genetic_algo_config = genetic_algo_config

    def run(self, population: Population) -> list[Observation]:
        population_development = PopulationDevelopment.defaulter()
        best_observations = list()

        while not self.stop_condition.stop(population_development):

            # perform mutation
            population, improvement_ratio = self.mutate(population)

            # perform crossover
            population = self.cross_over(population)

            # perform selection
            population = self.select.select(population)

            # updates
            population_development.update(population, improvement_ratio)
            self.genetic_algo_config.update(population_development)

            if not population_development.n_iter % self.genetic_algo_config.verbose_step:
                best_observations.append(population.best_observation())

            print(f'iter: {population_development.n_iter}, best:{population_development.best_score[-1]}')

        best_observations.append(population.best_observation())
        return best_observations

    def mutate(self, population):
        return Mutator(self.genetic_algo_config.mutation_proba).execute(population)

    def cross_over(self, population):
        return CrossOver(self.genetic_algo_config.crossover_proba).execute(population)
