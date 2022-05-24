# finding minimum of f(x,y,z) = x^2 + y^2 + 2z^2 (parabola)

import numpy as np

# params
from src.domain.algorithm import GeneticAlgorithm
from src.domain.population import Observation, Population
from src.domain.selector import SimpleSelector
from src.domain.stop_condition import StopConditionSimple

N_ITER = 1000
POPULATION_SIZE = 100
OBSERVATION_LENGTH = 3
CROSS_OVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.2

INITIAL_LEFT = -50
INITIAL_RIGHT = 50

INITIAL_POPULATION = np.random.uniform(INITIAL_LEFT, INITIAL_RIGHT, (POPULATION_SIZE, OBSERVATION_LENGTH))


class ParabolaObservation(Observation):

    def __init__(self, value: np.array):
        super().__init__(value)

    def evaluate(self) -> float:
        return -np.sum(self.value[0] ** 2 + self.value[1] ** 2 + 2 * self.value[2] ** 2)

    def mutate(self):
        return ParabolaObservation(self.value + np.random.uniform())

    def crossover(self, other):
        return ParabolaObservation((self.value + other.value) // 2)


def solve() -> Observation:
    ga = GeneticAlgorithm(
        StopConditionSimple(N_ITER),
        SimpleSelector(POPULATION_SIZE),
        CROSS_OVER_PROBABILITY,
        MUTATION_PROBABILITY)

    population = Population([ParabolaObservation(obs) for obs in INITIAL_POPULATION])

    return ga.run(population)


def main():
    print(f'Minimum of f(x,y,z) = x^2 + y^2 + 2z^2 is: {solve().value}')


if __name__ == '__main__':
    main()
