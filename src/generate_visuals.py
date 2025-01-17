import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tqdm import tqdm
from . import target_image
from .evolution import EvolutionSaveData
from .individual import paint_shapes

# target_path = sys.argv[1]
# run_name = sys.argv[2]

def create_frames(target_path, run_name):
    data = EvolutionSaveData.load(run_name)

    target = cv.imread(target_path)
    target_image.set_target_image(target, data.alpha)

    for (gen_index, gen_data) in tqdm(data.generations.items()):
        if gen_index >= 0:
            painted_image = paint_shapes(gen_data.best_shapes)
            frame_path = f"frames/{data.save_name}_{gen_index:05d}.png"
            cv.imwrite(frame_path, painted_image)

def create_graphs(target_path, run_name):
    data = EvolutionSaveData.load(run_name)

    target = cv.imread(target_path)
    target_image.set_target_image(target, data.alpha)

    min_fitness = []
    avg_fitness = []
    max_fitness = []
    shape_fitness = []

    for (gen_index, gen_data) in tqdm(data.generations.items()):
        min_fitness.append(gen_data.min_fitness)
        avg_fitness.append(gen_data.avg_fitness)
        max_fitness.append(gen_data.max_fitness)
        if gen_index % data.generations_per_shape == data.generations_per_shape - 1:
            shape_fitness.append(gen_data.min_fitness)

    def fitness(g):
        return data.generations[g].min_fitness

    generations_per_shape_weights = np.zeros(data.generations_per_shape)
    i = 0
    while i + data.generations_per_shape < len(min_fitness):
        total = fitness(i - 1) - fitness(i + data.generations_per_shape - 1)
        for j in range(data.generations_per_shape):
            # if fitness(i + j - 1) < fitness(i + j):
            #     print(i, j, fitness(i + j - 1), fitness(i + j), total, (fitness(i + j - 1) - fitness(i + j)) / total)
            generations_per_shape_weights[j] += (fitness(i + j - 1) - fitness(i + j)) / total
        i += data.generations_per_shape
    print(generations_per_shape_weights)
    generations_per_shape_weights /= generations_per_shape_weights.sum()

    num_generations = len(min_fitness)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(num_generations), min_fitness, label="Min Fitness", color="green")
    # ax1.plot(range(num_generations), avg_fitness, label="Avg Fitness", color="blue")
    # ax1.plot(range(num_generations), max_fitness, label="Max Fitness", color="red")
    ax1.set_yscale('log')
    ax1.set_xlabel("Generations")
    ax1.set_ylabel("Fitness")
    ax1.set_title("Fitness Over Generations (Log Scale)")
    ax1.legend()
    ax1.grid(True)

    num_shapes = len(shape_fitness)

    fig2, ax2 = plt.subplots()
    ax2.plot(range(num_shapes), shape_fitness, label="Min Fitness", color="green")
    # ax2.scatter(range(num_shapes), shape_fitness, color="red", zorder=4, s=1)
    ax2.set_yscale('log')
    ax2.set_xlabel("Number of Shapes")
    ax2.set_ylabel("Fitness")
    ax2.set_title("Fitness Over Number of Shapes (Log Scale)")
    ax2.legend()
    ax2.grid(True)


    fig3, ax3 = plt.subplots()
    ax3.bar(range(data.generations_per_shape), generations_per_shape_weights, color='blue')
    ax3.set_xlabel('Bar Number')
    ax3.set_ylabel('Value')
    ax3.set_title('Bar Plot of 15 Bars')

    return (fig1, fig2, fig3)

