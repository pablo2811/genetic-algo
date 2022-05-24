from abc import ABC, abstractmethod
from dataclasses import dataclass


class Observation(ABC):

    @abstractmethod
    def evaluate(self) -> float:
        pass

    @abstractmethod
    def mutate(self) -> Observation:
        pass

    @abstractmethod
    def crossover(self, observation: Observation) -> Observation:
        pass


class Population:

    def __init__(self, observations: list[Observation]):
        self.observations = observations

    def average_score(self) -> float:
        return sum([obs.evaluate() for obs in self.observations]) / (len(self.observations))

    def best_score(self) -> float:
        return max([obs.evaluate() for obs in self.observations])

    def best_observation(self) -> Observation:
        with_scores = sorted([(obs.evaluate(), obs) for obs in self.observations], key=lambda x: x[0])

        return with_scores[-1][1]


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


class StopCondition(ABC):

    @abstractmethod
    def stop(self, population_development: PopulationDevelopment) -> bool:
        pass


class PopulationTransformer(ABC):

    @abstractmethod
    def execute(self, population: Population) -> Population:
        pass


class GeneticAlgorithm:
    def __init__(self,
                 stop_condition: StopCondition,
                 cross_over: PopulationTransformer,
                 mutate: PopulationTransformer,
                 select: PopulationTransformer
                 ):
        self.select = select
        self.mutate = mutate
        self.cross_over = cross_over
        self.stop_condition = stop_condition

    def run(self, population: Population) -> Observation:
        population_development = PopulationDevelopment.defaulter()

        while not self.stop_condition.stop(population_development):
            # perform crossover
            population = self.cross_over.execute(population)

            # perform mutation
            population = self.mutate.execute(population)

            # perform selection
            population = self.select.execute(population)

            # update population development
            population_development.update(population)

        return population.best_observation()
