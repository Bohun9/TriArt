import numpy as np
import time
import cv2 as cv
import queue
import threading
import pickle
import copy
import target_image
from thread_local_data import thread_local
from individual import Individual

POPULATION_SIZE = 20
OFFSPRING_SIZE = 50
MAX_GENERATIONS = 10000
GENERATIONS_PER_SHAPE = 20
GENERATIONS_PER_SAVE = 100
MAX_NUMER_OF_SHAPES = 500
NUM_THREADS = 2
ESCAPE_KEY = 27

def generate_rng():
    seed = threading.get_ident() + int(time.time() * 1000)
    return np.random.default_rng(seed=seed)

thread_local.rng = generate_rng()

def evolve(image, image_name, population_path):
    target_image.image = image
    target_image.height, target_image.width, _ = image.shape

    starting_shapes = None
    num_triangles = 0
    if population_path is not None:
        with open(population_path, "rb") as f:
            starting_shapes = pickle.load(f)
            num_triangles = len(starting_shapes)
            print(f"loaded population from {population_path}")

    population = [Individual(triangles=copy.copy(starting_shapes)) for _ in range(POPULATION_SIZE)]
    cv.namedWindow("Best Individual")

    for generation_index in range(MAX_GENERATIONS):
        start_time = time.time()

        if generation_index % GENERATIONS_PER_SHAPE == 0 and num_triangles < MAX_NUMER_OF_SHAPES:
            num_triangles += 1
            for individual in population:
                individual.add_random_triangle()

        parent_probabilites = np.zeros(POPULATION_SIZE)
        for i in range(POPULATION_SIZE):
            parent_probabilites[i] = population[i].fitness
        parent_probabilites = parent_probabilites.max() - parent_probabilites + 1
        parent_probabilites /= parent_probabilites.sum()

        offspring = queue.Queue(maxsize=OFFSPRING_SIZE)

        def worker(num_children):
            thread_local.rng = generate_rng()
            for _ in range(num_children):
                parent = thread_local.rng.choice(population, p=parent_probabilites)
                child = parent.mutate()
                offspring.put(child)

        threads = []
        for i in range(NUM_THREADS):
            num_children = OFFSPRING_SIZE // NUM_THREADS + (1 if i < OFFSPRING_SIZE % NUM_THREADS else 0)
            thread = threading.Thread(target=worker, args=(num_children,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        candidates = population
        while not offspring.empty():
            candidates.append(offspring.get())
        assert (len(candidates) == POPULATION_SIZE + OFFSPRING_SIZE)
        candidates.sort(key=lambda x: x.fitness)
        population = candidates[:POPULATION_SIZE]

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"gen={generation_index} best={population[0].fitness:.3e} time={elapsed_time:.3f}sec size={num_triangles}")

        if generation_index % GENERATIONS_PER_SAVE == 0:
            mantissa, exponent = f"{population[0].fitness:.2e}".split("e")
            fitness_name = f"e{exponent.replace('+', '')}_m{mantissa.replace('.', '-')}"
            file_name = f"{image_name}_{fitness_name}.pkl"
            file_path = f"saved/{file_name}"
            with open(file_path, "wb") as f:
                pickle.dump(population[0].triangles, f)
            print(f"saved {file_name}")

        # painted_image = population[0].paint(background=target_image)
        painted_image = population[0].paint()
        cv.imshow("Best Individual", painted_image)
        key = cv.waitKey(300)
        if key == ESCAPE_KEY:
            break

cv.destroyAllWindows()
