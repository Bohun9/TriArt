import copy
import numpy as np
import cv2 as cv
from . import image
from . import target_image
from .thread_local_data import thread_local
from .bounding_box import BoundingBox
from .triangle import Triangle

ASSERTIONS = False # works only with custom rasterizer
PROBABILITY_MOVE_TRIANGLE = 0.98
PROBABILITY_REPLACE_TRIANGLE = 0.2

def paint(triangles, background=None):
    return image.paint_triangles(triangles, target_image.height, target_image.width, background=background)

def compute_fitness(painted_image):
    return image.compute_squared_error(painted_image, target_image.image)

class Individual:
    def __init__(self, triangles=None, fitness=None):
        self.triangles = triangles if triangles is not None else []
        self.fitness = fitness if fitness is not None else self.compute_fitness()
        if ASSERTIONS:
            assert (self.fitness == self.compute_fitness())

    def paint(self, background=None):
        return paint(self.triangles, background)

    def compute_fitness(self):
        return compute_fitness(self.paint())

    def add_random_triangle(self, focus_mode, target_pixels):
        mutated_triangles = copy.copy(self.triangles)

        best = (1e18, None)
        for _ in range(5):
            new_triangle = Triangle.random_triangle(focus_mode, target_pixels)
            mutated_triangles.append(new_triangle)
            new_fitness = self.recompute_fitness_from_parent(mutated_triangles, len(self.triangles))
            new_fitness = new_fitness if new_fitness is not None else compute_fitness(paint(mutated_triangles))
            if new_fitness < best[0]:
                best = (new_fitness, new_triangle)
            mutated_triangles.pop()

        self.triangles.append(best[1])
        self.fitness = best[0]

    @staticmethod
    def draw_index(size):
        assert size > 0
        p = np.array([i + 1 for i in range(size)], dtype=np.float64)
        p[-1] += 2.0 * p.sum() / 3.0
        p /= p.sum()
        return thread_local.rng.choice(size, p=p)

    def mutate(self, focus_mode, target_pixels):
        # triangles are shared in the memory between individuals
        mutated_triangles = copy.copy(self.triangles)
        index = Individual.draw_index(len(self.triangles))
        mutated_triangles[index] = copy.deepcopy(mutated_triangles[index])
        if thread_local.rng.random() < PROBABILITY_MOVE_TRIANGLE:
            if thread_local.rng.random() < PROBABILITY_REPLACE_TRIANGLE:
                mutated_triangles[index] = Triangle.random_triangle(focus_mode, target_pixels)
            else:
                mutated_triangles[index].adjust_vertices(focus_mode, target_pixels)
                if thread_local.rng.random() < 0.6:
                    mutated_triangles[index].change_color(focus_mode, target_pixels)
        else:
            mutated_triangles[index].change_color(focus_mode, target_pixels)
        new_fitness = self.recompute_fitness_from_parent(mutated_triangles, index)
        return Individual(triangles=mutated_triangles, fitness=new_fitness)

    def recompute_fitness_from_parent(self, mutated_triangles, index):
        bounding_box = self.triangles[index].bounding_box() if index < len(self.triangles) else mutated_triangles[index].bounding_box()
        bounding_box.unite(mutated_triangles[index].bounding_box())
        bounding_box.intersect(BoundingBox((0, 0), (target_image.width, target_image.height)))

        if bounding_box.is_empty():
            return self.fitness

        if 4 * bounding_box.area() > target_image.height * target_image.width:
            return None

        def compute_bounded_fitness(triangles):
            img = image.paint_triangles(triangles, bounding_box.height(), bounding_box.width(), bounding_box.corner1())
            bounded_fitness = image.compute_squared_error(img, bounding_box.get_region_of_image(target_image.image))
            return bounded_fitness, img

        bounded_old_fitness, old_triangle_area = compute_bounded_fitness(self.triangles)
        bounded_new_fitness, _ = compute_bounded_fitness(mutated_triangles)
        new_fitness = self.fitness + bounded_new_fitness - bounded_old_fitness 

        return new_fitness
