import numpy as np
import time
import cv2 as cv
import queue
import threading
import multiprocessing
import pickle
import copy
import sys
from . import target_image
from .thread_local_data import thread_local
from .individual import Individual
from .image import compute_target_pixels, compute_target_pixels2, compute_target_pixels3, compute_target_pixels4, absolute_difference, paint_points, paint_edges_of_triangles

GENERATIONS_PER_SHAPE = 15
GENERATIONS_PER_SAVE = 100
MAX_NUMER_OF_SHAPES = 500
NUM_WORKERS = 4
ESCAPE_KEY = 27

class GenerationSaveData:
    def __init__(self, min_fitness, avg_fitness, max_fitness, best_shapes, avg_last_triangle_size, time):
        self.min_fitness = min_fitness
        self.avg_fitness = avg_fitness
        self.max_fitness = max_fitness
        self.best_shapes = best_shapes
        self.avg_last_triangle_size = avg_last_triangle_size
        self.time = time

class EvolutionSaveData:
    def __init__(self, save_name, alpha):
        self.save_name = save_name
        self.alpha = alpha
        self.generations_per_shape = GENERATIONS_PER_SHAPE
        self.generations = {}

    @staticmethod
    def load(save_name):
        file_path = f"saved/{save_name}.pkl"
        res = None
        with open(file_path, "rb") as f:
            res = pickle.load(f)
        return res

    def save(self):
        file_path = f"saved/{self.save_name}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        print(f"saved {file_path}")

def generate_rng():
    seed = threading.get_ident() + int(time.time() * 1000)
    return np.random.default_rng(seed=seed)

thread_local.rng = generate_rng()

focus_mode = False
target_pixels = None

def process_worker(parents):
    thread_local.rng = generate_rng()
    children = []
    for parent in parents:
        child = parent.mutate(focus_mode, target_pixels)
        children.append(child)
    return children

def evolve(image, save_name, generation_index=None, display_mode="image", use_threads=False, alpha=0.6, population_size=20, offspring_size=60, max_generations=None):
    global focus_mode, target_pixels

    print(f"starting evolution save_name={save_name} generation_id={generation_index} display_mode={display_mode} alpha={alpha}")

    target_image.set_target_image(image, alpha)

    evolution_save_data = None
    starting_shapes = None
    num_triangles = 0
    if generation_index is None:
        generation_index = -1
        evolution_save_data = EvolutionSaveData(save_name=save_name, alpha=alpha)
    else:
        evolution_save_data = EvolutionSaveData.load(save_name)
        print("loaded population")
        if generation_index not in evolution_save_data.generations:
            sys.exit("invalid generation index")
        if alpha != evolution_save_data.alpha:
            sys.exit("alpha parameter should be the same as in the loaded run")
        starting_shapes = evolution_save_data.generations[generation_index].best_shapes
        num_triangles = len(starting_shapes)

    population = [Individual(triangles=copy.copy(starting_shapes)) for _ in range(population_size)]
    cv.namedWindow("Best Individual")

    if generation_index == -1:
        evolution_save_data.generations[generation_index] = GenerationSaveData(
                min_fitness=population[0].fitness,
                avg_fitness=population[0].fitness,
                max_fitness=population[0].fitness,
                best_shapes=population[0].triangles,
                avg_last_triangle_size=0,
                time=0
        )
    generation_index += 1

    max_generations = max_generations if max_generations is not None else MAX_NUMER_OF_SHAPES * GENERATIONS_PER_SHAPE

    while generation_index < max_generations:
        start_time = time.time()

        focus_mode = num_triangles > 40
        if generation_index % GENERATIONS_PER_SHAPE == 0:
            target_pixels = compute_target_pixels4(population[0].paint(), target_image.image)[:200]

        if generation_index % GENERATIONS_PER_SHAPE == 0 and num_triangles < MAX_NUMER_OF_SHAPES:
            num_triangles += 1
            for individual in population:
                individual.add_random_triangle(focus_mode, target_pixels)

        parent_probabilites = np.zeros(population_size)
        for i in range(population_size):
            parent_probabilites[i] = population[i].fitness
        parent_probabilites = parent_probabilites.max() - parent_probabilites + 1
        parent_probabilites /= parent_probabilites.sum()

        parents = np.array([thread_local.rng.choice(population, p=parent_probabilites) for  _ in range(offspring_size)])
        parents_chunks = []
        chunk_start = 0
        for i in range(NUM_WORKERS):
            size = offspring_size // NUM_WORKERS + (1 if i < offspring_size % NUM_WORKERS else 0)
            parents_chunks.append(parents[chunk_start:chunk_start + size])
            chunk_start += size

        last_triangles_offspring = []

        if use_threads:
            offspring = queue.Queue(maxsize=offspring_size)

            def thread_worker(parents):
                thread_local.rng = generate_rng()
                for parent in parents:
                    child = parent.mutate(focus_mode, target_pixels)
                    offspring.put(child)

            threads = []
            for i in range(NUM_WORKERS):
                num_children = offspring_size // NUM_WORKERS + (1 if i < offspring_size % NUM_WORKERS else 0)
                thread = threading.Thread(target=thread_worker, args=(parents_chunks[i],))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            while not offspring.empty():
                child = offspring.get()
                population.append(child)
                last_triangles_offspring.append(child.triangles[-1])
        else:
            with multiprocessing.Pool(NUM_WORKERS) as pool:
                offspring = pool.map(process_worker, parents_chunks)

            for group in offspring: 
                population.extend(group)
                for ind in group:
                    last_triangles_offspring.append(ind.triangles[-1])

        assert (len(population) == population_size + offspring_size)
        population.sort(key=lambda x: x.fitness)
        population = population[:population_size]

        end_time = time.time()
        elapsed_time = end_time - start_time

        evolution_save_data.generations[generation_index] = GenerationSaveData(
                min_fitness=population[0].fitness,
                avg_fitness=np.average(np.array([ind.fitness for ind in population])),
                max_fitness=population[-1].fitness,
                best_shapes=population[0].triangles, 
                avg_last_triangle_size=np.average(np.array([ind.triangles[-1].area() for ind in population])),
                time=elapsed_time
        )

        start_time = time.time()

        if display_mode == "image":
            cv.imshow("Best Individual", population[0].paint())
        elif display_mode == "details" and target_pixels is not None:
            painted_image = population[0].paint()
            target_pixels_image = paint_points(target_pixels, target_image.image)
            last_triangles_population = []
            for i in range(population_size):
                last_triangles_population.append(population[i].triangles[-1])
            recently_added_trangles = population[0].triangles[-4:-1]
            paint_edges_of_triangles(recently_added_trangles, painted_image, (0, 255, 0))
            paint_edges_of_triangles(last_triangles_offspring, painted_image, (255, 0, 0))
            paint_edges_of_triangles(last_triangles_population, painted_image, (0, 0, 255))
            cv.imshow("Best Individual", np.hstack((painted_image, target_pixels_image)))

        end_time = time.time()
        debug_time = end_time - start_time

        print(f"gen={generation_index} best={population[0].fitness:.3e} time={elapsed_time:.3f} debug_time={debug_time:.3f}sec size={num_triangles} max_prob={parent_probabilites[0]:2f}")

        key = cv.waitKey(20)
        if key == ESCAPE_KEY:
            break

        generation_index += 1

    evolution_save_data.save()
    cv.destroyAllWindows()
