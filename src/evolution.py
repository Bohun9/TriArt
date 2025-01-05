import numpy as np
import time
import cv2 as cv
import queue
import threading
import multiprocessing
import pickle
import copy
from . import target_image
from .thread_local_data import thread_local
from .individual import Individual
from .image import compute_target_pixels, compute_target_pixels2, compute_target_pixels3, compute_target_pixels4, absolute_difference, paint_points, paint_triangles, paint_edges_of_triangles

GENERATIONS_PER_SHAPE = 15
GENERATIONS_PER_SAVE = 100
MAX_NUMER_OF_SHAPES = 500
NUM_WORKERS = 4
ESCAPE_KEY = 27

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

def evolve(image, save_name, generation_index=None, record=False, save_all=False, debug=False, use_threads=False, alpha=0.6, population_size=20, offspring_size=60, max_generations=None):
    global focus_mode, target_pixels

    print(f"starting evolution save_name={save_name} population_id={generation_index} record={record} save_all={save_all} alpha={alpha}")

    target_image.set_target_image(image, alpha)

    starting_shapes = None
    num_triangles = 0
    if generation_index is None:
        generation_index = -1
    else:
        population_path = f"saved/{save_name}_{generation_index:05d}.pkl"
        with open(population_path, "rb") as f:
            starting_shapes = pickle.load(f)
            num_triangles = len(starting_shapes)
            print(f"loaded population from {population_path}")
    generation_index += 1

    population = [Individual(triangles=copy.copy(starting_shapes)) for _ in range(population_size)]
    cv.namedWindow("Best Individual")

    max_generations = max_generations if max_generations is not None else MAX_NUMER_OF_SHAPES * GENERATIONS_PER_SHAPE

    while generation_index < max_generations:
        start_time = time.time()

        focus_mode = num_triangles > 40
        if generation_index % GENERATIONS_PER_SHAPE == 0:
            target_pixels = compute_target_pixels4(population[0].paint(), target_image.image)[:100]

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
                population.append(offspring.get())
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

        print(f"gen={generation_index} best={population[0].fitness:.3e} time={elapsed_time:.3f}sec size={num_triangles} max_prob={parent_probabilites[0]:2f}")

        if save_all or generation_index % GENERATIONS_PER_SAVE == 0:
            file_name = f"{save_name}_{generation_index:05d}.pkl"
            file_path = f"saved/{file_name}"
            with open(file_path, "wb") as f:
                pickle.dump(population[0].triangles, f)
            print(f"saved {file_name}")

        painted_image = population[0].paint()

        if record:
            file_name = f"frames/{save_name}_{generation_index:05d}.png"
            cv.imwrite(file_name, painted_image)

        if debug and target_pixels is not None:
            target_pixels_image = paint_points(target_pixels, target_image.image)
            last_triangles_population = []
            for i in range(population_size):
                last_triangles_population.append(population[i].triangles[-1])
            recently_added_trangles = population[0].triangles[-4:-1]
            paint_edges_of_triangles(recently_added_trangles, painted_image, (0, 255, 0))
            paint_edges_of_triangles(last_triangles_offspring, painted_image, (255, 0, 0))
            paint_edges_of_triangles(last_triangles_population, painted_image, (0, 0, 255))
            cv.imshow("Best Individual", np.hstack((painted_image, target_pixels_image)))
        else:
            cv.imshow("Best Individual", painted_image)

        key = cv.waitKey(300)
        if key == ESCAPE_KEY:
            break

        generation_index += 1

cv.destroyAllWindows()
