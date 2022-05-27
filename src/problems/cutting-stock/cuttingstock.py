import numpy as np

from src.domain.algorithm import GeneticAlgorithm
from src.domain.population import Observation, Population
from src.domain.selector import SimpleSelector
from src.domain.stop_condition import StopConditionSimple
from util import get_random_point_in_circle

N_ITER = 200
POPULATION_SIZE = 50
CROSS_OVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.2
MAX_ATTEMPTS_ONE_OBS = 50

INTERSECTS = 0
ABOVE = 1
BELOW = -1


class Rectangle:
    def __init__(self, height: float, width: float, value: float):
        self.value = value
        self.width = width
        self.height = height


class PlacedRectangle(Rectangle):
    def __init__(self, x, y, height: float, width: float, value: float):
        super().__init__(height, width, value)
        # top-left
        self.y = y
        self.x = x

    def overlaps(self, other):
        x_corner_self, y_corner_self = self.other_corner()
        x_corner_other, y_corner_other = other.other_corner()

        return PlacedRectangle.overlap(self.x, x_corner_self, other.x, x_corner_other) and PlacedRectangle.overlap(
            y_corner_self, self.y, y_corner_other, other.y)

    def within_circle(self, r: float, x=0, y=0):

        within_circle = True
        for corner in self.all_corners():
            within_circle = within_circle and PlacedRectangle.within_circle_point(corner, r, x, y)

        return within_circle

    def line_relation(self, a: float, b: float = 0):
        vals = [a * corner[0] + b for corner in self.all_corners()]
        if vals[0] > 0 and all([val > 0 for val in vals]):
            return ABOVE

        elif vals[0] < 0 and all([val < 0 for val in vals]):
            return BELOW

        return INTERSECTS

    def free_of_overlaps(self, rectangles):
        for existing_rectangle in rectangles:
            if self.overlaps(existing_rectangle):
                return False

        return True

    def all_corners(self):
        return [(self.x, self.y), self.other_corner(), (self.x, self.y - self.height),
                (self.x + self.width, self.y)]

    def other_corner(self):
        return self.x + self.width, self.y - self.height

    @staticmethod
    def within_circle_point(point: tuple, r: float, x=0, y=0):
        return (point[0] - x) ** 2 + (point[1] - y) ** 2 <= r ** 2

    @staticmethod
    def overlap(a1, a2, b1, b2):
        """ are two intervals overlapping """
        if a1 < b1:
            return not (a2 <= b1)
        else:
            return not (b2 <= a1)


class CuttingStockObservation(Observation):

    def __init__(self, rectangles: list[PlacedRectangle], available_rectangles: list[Rectangle], r: float):
        self.r = r
        self.available_rectangles = available_rectangles
        self.rectangles = rectangles

    def evaluate(self) -> float:
        return self.evaluate_subset(self.rectangles)

    # TODO: different mutation strategies
    def mutate(self):
        return self.cut_out_and_replace_mutate_()

    def crossover(self, other):
        return self.combine_best_halves(other)

    def cut_out_and_replace_mutate_(self):
        cut_middle = get_random_point_in_circle(self.r)
        r_cut = np.random.uniform(0, self.r - np.linalg.norm(cut_middle))
        rectangles = list(
            filter(lambda rect: not rect.within_circle(r_cut, cut_middle[0], cut_middle[1]), self.rectangles)
        )
        self.add_rectangles(self.available_rectangles, self.r, rectangles)

        return CuttingStockObservation(rectangles, self.available_rectangles, self.r)

    def combine_best_halves(self, other):

        best = 0
        best_rectangles = []

        for a in range(10, 1000, 10):
            for b in (a, -a):
                self_split = self.split(b)
                other_split = other.split(b)
                opt1, opt2 = self_split[ABOVE] + other_split[BELOW], self_split[BELOW] + other_split[ABOVE]
                if self.evaluate_subset(opt1) > best:
                    best_rectangles = opt1
                if self.evaluate_subset(opt2) > best:
                    best_rectangles = opt2

        return CuttingStockObservation(best_rectangles, self.available_rectangles, self.r)

    def split(self, a):

        grouped = {ABOVE: [], BELOW: [], INTERSECTS: []}
        for rect in self.rectangles:
            grouped[rect.line_relation(a)].append(rect)

        return grouped

    @staticmethod
    def add_rectangles(available_rectangles, r, rectangles, x=0, y=0):
        for _ in range(MAX_ATTEMPTS_ONE_OBS):
            for rect in available_rectangles:
                point = get_random_point_in_circle(r, x, y)
                rectangle = PlacedRectangle(point[0], point[1], rect.height, rect.width, rect.value)
                if rectangle.free_of_overlaps(rectangles) and rectangle.within_circle(r, x, y):
                    rectangles.append(rectangle)

    @staticmethod
    def evaluate_subset(rectangles):
        return sum(rect.value for rect in rectangles)


def initialize_population(available_rectangles: list[Rectangle], r: float):
    return [initialize_observation(available_rectangles, r) for _ in range(POPULATION_SIZE)]


def initialize_observation(available_rectangles: list[Rectangle], r: float) -> CuttingStockObservation:
    rectangles = list()
    CuttingStockObservation.add_rectangles(available_rectangles, r, rectangles)

    return CuttingStockObservation(rectangles, available_rectangles, r)


def solve() -> Observation:
    ga = GeneticAlgorithm(
        StopConditionSimple(N_ITER),
        SimpleSelector(POPULATION_SIZE),
        CROSS_OVER_PROBABILITY,
        MUTATION_PROBABILITY)

    init_pop = Population(initialize_population([Rectangle(1, 1, 1), Rectangle(1, 2, 2)], 20))

    return ga.run(init_pop, list(range(10, 200, 5)))


def main():
    solution = solve()
    print(f'Value found is: {solution.evaluate()}')


if __name__ == '__main__':
    main()
