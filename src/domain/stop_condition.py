import time
from abc import abstractmethod, ABC

from src.domain.population import PopulationDevelopment


class StopCondition(ABC):

    @abstractmethod
    def stop(self, population_development: PopulationDevelopment) -> bool:
        pass


class StopConditionCombined(StopCondition):

    def __init__(self, conditions: list[StopCondition]):
        self.conditions = conditions

    def stop(self, population_development: PopulationDevelopment) -> bool:
        should_stop = False

        for cond in self.conditions:
            should_stop = should_stop or cond.stop(population_development)

        return should_stop


class StopConditionSimple(StopCondition):

    def __init__(self, n_iter: int):
        self.n_iter = n_iter

    def stop(self, population_development: PopulationDevelopment) -> bool:
        return population_development.n_iter > self.n_iter


class StopConditionTime(StopCondition):
    def __init__(self, max_time: float):
        self.max_time = max_time

    def stop(self, population_development: PopulationDevelopment) -> bool:
        return time.time() - population_development.start_time > self.max_time
