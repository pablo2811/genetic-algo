import time
from abc import abstractmethod, ABC
from dataclasses import dataclass


class Observation(ABC):

    @abstractmethod
    def evaluate(self) -> float:
        pass

    @abstractmethod
    def mutate(self):
        pass

    @abstractmethod
    def crossover(self, observation):
        pass


class Population:

    def __init__(self, observations: list[Observation]):
        self.observations = observations

    def size(self):
        return len(self.observations)

    def average_score(self) -> float:
        return sum([obs.evaluate() for obs in self.observations]) / (len(self.observations))

    def best_score(self) -> float:
        return max([obs.evaluate() for obs in self.observations])

    def n_best_observations(self, n) -> list[Observation]:
        with_scores = self.with_scores_()

        return [with_scores[i][1] for i in range(n)]

    def best_observation(self) -> Observation:
        with_scores = self.with_scores_()

        return with_scores[0][1]

    def with_scores_(self):
        return sorted([(obs.evaluate(), obs) for obs in self.observations], key=lambda x: x[0], reverse=True)


@dataclass
class PopulationDevelopment:
    n_iter: int
    average_score: float
    best_score: list[float]
    start_time: float
    improved_mutations: list[bool]

    def update(self, population: Population, mutation_improvement_ratio):
        self.n_iter += 1
        self.average_score = population.average_score()
        self.best_score.append(population.best_score())
        self.improved_mutations.append(mutation_improvement_ratio > 0.2)

    @staticmethod
    def defaulter():
        return PopulationDevelopment(0, float("+inf"), [], time.time(), [])
