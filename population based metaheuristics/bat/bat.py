import numpy as np
import math
import os
import random
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from Natural_algorithms.bat.BatAlgorithmBase import BatAlgorithm
from Natural_algorithms.gwo.gwo import plot_rastrigin_contour


def fitness_rastrigin(dim, position):
    fitness_value = 0.0
    for i in range(dim):
        xi = position[i]
        fitness_value += ((xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10)
    return fitness_value


def run_bat(population, max_iteration, dim, fitness, minx=-5.12, maxx=5.12):
    bat = BatAlgorithm(dim, population, max_iteration, 2.0, 0.05, 0.1, 2.0, minx, maxx, fitness)
    bat.move_bat()
    results = bat.get_best_solutions()
    print("Best solution:", results[0], " fitness:", results[1])
    return results


def main():
    # parameters
    dim = 3
    max_iteration = 500
    populations = [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
    computed_solutions = []
    computed_fitness = []
    # run bat algorithm
    for population in populations:
        print("Population:", population)
        results = run_bat(population, max_iteration, dim, fitness_rastrigin)
        computed_solutions.append(results[0])
        computed_fitness.append(results[1])
    # plot results of bat algorithm on rastrigin function contour
    plot_rastrigin_contour(computed_solutions, "3")
    # save population, solutions and fitness in csv file in an output folder
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    np.savetxt("outputs/bat_rastrigin_3.csv", np.column_stack((populations, computed_solutions, computed_fitness)),
               delimiter=",", fmt='%s')


if __name__ == "__main__":
    main()
