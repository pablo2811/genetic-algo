import numpy as np

from src.domain.population import Observation
from util import get_random_point_in_circle

INTERSECTS = 0
LEFT = 1
RIGHT = -1


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

    def vertical_slice(self, val: float):
        if val < self.x:
            return LEFT

        if self.x + self.width < val:
            return RIGHT

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

    def __init__(self, available_rectangles: list[Rectangle], r: float, attempts_fill: int = 1,
                 rectangles: list[PlacedRectangle] = []):
        self.r = r
        self.available_rectangles = available_rectangles
        self.rectangles = rectangles
        self.attempts_fill = attempts_fill

    def evaluate(self) -> float:
        return self.evaluate_subset(self.rectangles)

    def mutate(self):
        return self.mutate_singular()

    def mutate_circle(self):
        center, r_cut = self.micro_circle()

        if np.random.uniform() < 0.75:
            new_rectangles = self.fill_with_rectangles(center, r_cut)
        else:
            new_rectangles = self.cut_out(center, r_cut)

        return CuttingStockObservation(self.available_rectangles, self.r, self.attempts_fill, new_rectangles)

    def mutate_singular(self):

        if np.random.uniform() < 0.75:
            new_rectangles = self.fill_with_rectangles((0, 0), self.r)
        else:
            new_rectangles = self.rectangles.copy()
            if len(self.rectangles) > 0:
                ind_delete = np.random.randint(0, len(self.rectangles))
                new_rectangles.pop(ind_delete)

        return CuttingStockObservation(self.available_rectangles, self.r, self.attempts_fill, new_rectangles)

    def crossover(self, other):
        return self.combine_best_halves(other)

    def fill_with_rectangles(self, center: tuple, r: float):
        return self.rectangles + self.add_rectangles(center, r)

    def micro_circle(self):
        cut_middle = get_random_point_in_circle(self.r)
        r_cut = np.random.uniform(0, self.r - np.linalg.norm(cut_middle))
        return cut_middle, r_cut

    def cut_out(self, center: tuple, r: float):
        return list(
            filter(lambda rect: not rect.within_circle(r, center[0], center[1]), self.rectangles)
        )

    def combine_best_halves(self, other):

        best = 0
        best_rectangles = []

        for cut in [-self.r / 2, 0, self.r / 2]:
            self_split, other_split = self.split(cut), other.split(cut)
            self_split_eval, other_split_eval = self.evaluate_all(self_split), self.evaluate_all(other_split)

            a = self_split_eval[LEFT] + other_split_eval[RIGHT]
            b = self_split_eval[RIGHT] + other_split_eval[LEFT]

            best, best_rectangles = self.update_if_best(best, best_rectangles, a, b, self_split, other_split)
            best, best_rectangles = self.update_if_best(best, best_rectangles, b, a, other_split, self_split)

        return CuttingStockObservation(self.available_rectangles, self.r,
                                       (self.attempts_fill + other.attempts_fill) // 2, best_rectangles)

    def split(self, a):

        grouped = {LEFT: [], RIGHT: [], INTERSECTS: []}
        for rect in self.rectangles:
            grouped[rect.vertical_slice(a)].append(rect)

        return grouped

    def add_rectangles(self, center: tuple, r: float):
        new_rectangles = list()

        x, y = center
        sorted_rectangles = sorted(self.available_rectangles, key=lambda ar: ar.value / (ar.width * ar.height),
                                   reverse=True)
        for _ in range(self.attempts_fill):
            for rect in sorted_rectangles:
                point = get_random_point_in_circle(r, x, y)
                rectangle = PlacedRectangle(point[0], point[1], rect.height, rect.width, rect.value)
                if rectangle.free_of_overlaps(self.rectangles) and rectangle.free_of_overlaps(
                        new_rectangles) and rectangle.within_circle(r, x, y):
                    new_rectangles.append(rectangle)
                    break

        return new_rectangles

    @staticmethod
    def evaluate_subset(rectangles):
        return sum(rect.value for rect in rectangles)

    @staticmethod
    def evaluate_all(dict_of_subsets):
        return {k: CuttingStockObservation.evaluate_subset(dict_of_subsets[k]) for k in dict_of_subsets}

    @staticmethod
    def update_if_best(best, best_rectangles, a, b, self_split, other_split):
        if a > b and a > best:
            best = a
            best_rectangles = self_split[LEFT] + other_split[RIGHT]
        return best, best_rectangles
