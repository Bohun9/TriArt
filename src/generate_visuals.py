import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tqdm import tqdm
from . import target_image
from .evolution import EvolutionSaveData
from .individual import paint_shapes

def create_frames(target_path, run_name):
    data = EvolutionSaveData.load(run_name)

    target = cv.imread(target_path)
    target_image.set_target_image(target, data.alpha)

    for (gen_index, gen_data) in tqdm(data.generations.items()):
        if gen_index >= 0:
            painted_image = paint_shapes(gen_data.best_shapes)
            frame_path = f"frames/{data.save_name}_{gen_index:05d}.png"
            cv.imwrite(frame_path, painted_image)

class Visualizer:
    def __init__(self, run_name):
        data = EvolutionSaveData.load(run_name)

        self.data = data
        self.min_fitness = []
        self.avg_fitness = []
        self.max_fitness = []
        self.shape_fitness = []
        self.generation_times = []
        self.shape_times = []
        self.avg_shape_sizes = []

        for (gen_index, gen_data) in data.generations.items():
            self.min_fitness.append(gen_data.min_fitness)
            self.avg_fitness.append(gen_data.avg_fitness)
            self.max_fitness.append(gen_data.max_fitness)
            self.generation_times.append(gen_data.time)
            if gen_index % data.generations_per_shape == data.generations_per_shape - 1:
                self.shape_fitness.append(gen_data.min_fitness)
                self.avg_shape_sizes.append(gen_data.avg_last_triangle_size)

        def fitness(g):
            return data.generations[g].min_fitness

        self.generations_per_shape_weights = np.zeros(data.generations_per_shape)
        i = 0
        while i + data.generations_per_shape < len(self.min_fitness):
            time_sum = 0
            total = fitness(i - 1) - fitness(i + data.generations_per_shape - 1)
            for j in range(data.generations_per_shape):
                self.generations_per_shape_weights[j] += (fitness(i + j - 1) - fitness(i + j)) / total
                time_sum += data.generations[i + j].time
            self.shape_times.append(time_sum)
            i += data.generations_per_shape
        self.generations_per_shape_weights /= self.generations_per_shape_weights.sum()

    def draw_fitness_over_generations(self):
        fig, ax = plt.subplots()
        ax.plot(self.min_fitness, label="Min Fitness", color="green")
        ax.set_yscale('log')
        ax.set_xlabel("Generations")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness Over Generations (Log Scale)")
        ax.legend()
        ax.grid(True)

    def draw_changes_of_fitness(self):
        fig, ax = plt.subplots()
        fitness_diff = -np.diff(self.shape_fitness)
        batch_size = 7
        batched_diff = [np.mean(fitness_diff[i:i + batch_size]) for i in range(0, len(fitness_diff), batch_size)]
        ax.plot(batched_diff, marker='o', markersize=3, label=f"Average Δ (Batch Size={batch_size})")
        ax.set_yscale('log')
        ax.set_xlabel("Batch Id")
        ax.set_ylabel("Change in Fitness")
        ax.set_title("Consecutive Changes in Fitness Values Over the Number of Shapes")
        ax.legend()
        ax.grid(True)

    def draw_fitness_contribution(self):
        fig, ax = plt.subplots()
        ax.bar(range(self.data.generations_per_shape), self.generations_per_shape_weights, color='blue')
        ax.set_xlabel('Generation Id')
        ax.set_ylabel('Percent of Fitness Improvement')
        ax.set_title('Fitness Improvement Contribution per Shape Addition Generation')
        ax.grid(True)

    def draw_average_shape_area(self):
        fig, ax = plt.subplots()
        batch_size = 4
        batched_avg_shape_sizes = [np.mean(self.avg_shape_sizes[i:i + batch_size]) for i in range(0, len(self.avg_shape_sizes), batch_size)]
        ax.plot(batched_avg_shape_sizes, label=f"Average area (Batch Size={batch_size})")
        ax.set_yscale('log')
        ax.set_xlabel("Batch Id")
        ax.set_ylabel("Average Shape Area")
        ax.set_title("Average Shape Area of Added Triangle")
        ax.legend()
        ax.grid(True)

    def draw_time_over_generations(self):
        fig, ax = plt.subplots()
        accum_times = np.add.accumulate(self.generation_times) / 60
        ax.plot(accum_times)
        ax.set_xlabel("Generations")
        ax.set_ylabel("Time in Minutes")
        ax.set_title("Time Over Generations")
        ax.grid(True)

    def draw_time_per_shape(self):
        fig, ax = plt.subplots()
        batch_size = 4
        batched_shape_times = [np.mean(self.shape_times[i:i + batch_size]) for i in range(0, len(self.shape_times), batch_size)]
        ax.plot(self.shape_times)
        ax.set_xlabel("Number of Shapes Batch Id")
        ax.set_ylabel("Time in Seconds")
        ax.set_title("Time Per Shape (Batched)")
        ax.grid(True)

if __name__ == "__main__":
    target_path = sys.argv[1]
    run_name = sys.argv[2]

    create_frames(target_path, run_name)
