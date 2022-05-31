import numpy as np

# params
from src.domain.algorithm import GeneticAlgorithm
from src.domain.population import Observation, Population
from src.domain.selector import SimpleSelector, RouletteSelector
from src.domain.stop_condition import StopConditionSimple


# finding minimum of rastrigin function


N_ITER = 1000
POPULATION_SIZE = 10
OBSERVATION_LENGTH = 5
CROSS_OVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.2

INITIAL_LEFT = -5.12
INITIAL_RIGHT = 5.12

INITIAL_POPULATION = np.random.uniform(INITIAL_LEFT, INITIAL_RIGHT, (POPULATION_SIZE, OBSERVATION_LENGTH))


class RastriginObservation(Observation):

    def __init__(self, value: np.array):
        super().__init__(value)

    def evaluate(self) -> float:
        return -self.rastrigin_()

    def mutate(self):
        return RastriginObservation(self.value + np.random.uniform())

    def crossover(self, other):
        return RastriginObservation((self.value + other.value) // 2)

    def rastrigin_(self):
        x = np.asarray_chkfinite(self.value)
        n = len(x)
        return 10 * n + sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def solve() -> Observation:
    ga = GeneticAlgorithm(
        StopConditionSimple(N_ITER),
        RouletteSelector(POPULATION_SIZE),
        CROSS_OVER_PROBABILITY,
        MUTATION_PROBABILITY)

    population = Population([RastriginObservation(obs) for obs in INITIAL_POPULATION])

    return ga.run(population)[0]


def main():
    solution = solve()
    print(f'Minimum of rastrigin is: {solution.value} with value {solution.evaluate()}')


if __name__ == '__main__':
    main()
