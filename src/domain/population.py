from abc import abstractmethod, ABC
from dataclasses import dataclass


class Observation(ABC):

    def __init__(self, value):
        self.value = value

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
    best_score: float

    def update(self, population: Population):
        self.n_iter += 1
        self.average_score = population.average_score()
        self.best_score = population.best_score()

    @staticmethod
    def defaulter():
        return PopulationDevelopment(0, float("+inf"), float("+inf"))
