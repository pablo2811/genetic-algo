from src.domain.domain import StopCondition, PopulationDevelopment


class StopConditionImpl(StopCondition):

    def __init__(self, conditions: list[StopCondition]):
        self.conditions = conditions

    def stop(self, population_development: PopulationDevelopment) -> bool:
        should_stop = False

        for cond in self.conditions:
            should_stop = should_stop or cond.stop(population_development)

        return should_stop
