import copy
import numpy as np
import cv2 as cv
import image
from thread_local_data import thread_local
from bounding_box import BoundingBox
from triangle import Triangle
import target_image

ASSERTIONS = False
PROBABILITY_MOVE_TRIANGLE = 0.8
PROBABILITY_REPLACE_TRIANGLE = 0.2
EXPECTED_NUMBER_OF_SHAPES = 200

class Individual:
    def __init__(self, triangles=None, fitness=None):
        self.triangles = triangles if triangles is not None else []
        self.fitness = fitness if fitness is not None else self.compute_fitness()
        # if ASSERTIONS:
            # doesn't always work
            # assert (self.fitness == self.compute_fitness())

    def paint(self, background=None):
        return image.paint_triangles(self.triangles, target_image.height, target_image.width, background=background)

    def compute_fitness(self):
        return image.compute_squared_error(self.paint(), target_image.image)

    def add_random_triangle(self):
        self.triangles.append(Triangle.random_triangle())
        self.fitness = self.compute_fitness()

    @staticmethod
    def draw_index(size):
        assert size > 0
        p = np.array([i + 1 for i in range(size)], dtype=np.float64)
        p /= p.sum()
        return thread_local.rng.choice(size, p=p)

    def mutate(self):
        # triangles are shared in the memory between individuals
        mutated_triangles = copy.copy(self.triangles)
        index = Individual.draw_index(len(self.triangles))
        mutated_triangles[index] = copy.deepcopy(mutated_triangles[index])
        if thread_local.rng.random() < PROBABILITY_MOVE_TRIANGLE:
            if thread_local.rng.random() < PROBABILITY_REPLACE_TRIANGLE:
                mutated_triangles[index] = Triangle.random_triangle()
            else:
                mutated_triangles[index].adjust_vertices(len(self.triangles) / EXPECTED_NUMBER_OF_SHAPES)
                mutated_triangles[index].change_color()
        else:
            mutated_triangles[index].change_color()
        new_fitness = self.recompute_fitness_from_parent(mutated_triangles, index)
        return Individual(triangles=mutated_triangles, fitness=new_fitness)

    def recompute_fitness_from_parent(self, mutated_triangles, index):
        bounding_box = self.triangles[index].bounding_box()
        bounding_box.unite_in_place(mutated_triangles[index].bounding_box())
        bounding_box.intersect_in_place(BoundingBox((0, 0), (target_image.width, target_image.height)))

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

        # sometimes opencv draws different edges when canvas is shifted
        # if ASSERTIONS:
        #     img1 = self.paint()[c1[1]:c2[1], c1[0]:c2[0], :]
        #     img2 = old_triangle_area
        #     if np.array_equal(img1, img2) == False:
        #         cv.imwrite("debug.png", img1 - img2)
        #     assert (np.array_equal(img1, img2))

        return new_fitness
