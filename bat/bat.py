import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from Natural_algorithms.bat.BatAlgorithmBase import BatAlgorithm
from Natural_algorithms.gwo.gwo import plot_rastrigin_contour


def fitness_rastrigin(dim, position):
    fitness_value = 0.0
    for i in range (dim):
        xi = position[i]
        fitness_value += ((xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10)
    return fitness_value


def run_bat(population, max_iteration, dim, fitness, minx=-5.12, maxx=5.12):
    bat = BatAlgorithm(dim, population, max_iteration, 1.0, 0.9, 0.1, 2.0, minx, maxx, fitness)
    bat.move_bat()
    results = bat.get_best_solutions()
    print("Best solution:", results[0], " fitness:", results[1])
    return results


def main():
    # parameters
    dim = 3
    max_iteration = 400
    populations = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000]
    computed_solutions = []
    computed_fitness = []
    # run bat algorithm
    for population in populations:
        print("Population:", population)
        results = run_bat(population, max_iteration, dim, fitness_rastrigin)
        computed_solutions.append(results[0])
        computed_fitness.append(results[1])


if __name__ == "__main__":
    main()

