# implement the GWO algorithm for the optimization of the LEVY function
import copy

import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os


def fitness_rastrigin(position):
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += ((xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10)
    return fitness_value


class Wolf:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)
        self.position = [0.0 for j in range(dim)]
        # initialize position randomly within bounds of the search space [-5.12, 5.12]
        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)

        self.fitness = fitness(self.position)  # curr fitness of the wolf


def gwo(fitness, max_iteration, pop, dim, minx, maxx):
    rndm = random.Random(pop)
    population = [Wolf(fitness, dim, minx, maxx, i) for i in range(pop)]
    population = sorted(population, key=lambda temp: temp.fitness)
    alpha = copy.copy(population[0])
    beta = copy.copy(population[1])
    gamma = copy.copy(population[2])
    iter = 0
    while iter < max_iteration:
        if iter % 10 == 0 and iter > 0:
            print("iter:", iter, " alpha:", alpha.fitness)
        a = 2 - 5 * (2 / max_iteration)
        for i in range(pop):
            A1 = 2 * a * rndm.random() - a
            A2 = 2 * a * rndm.random() - a
            A3 = 2 * a * rndm.random() - a
            C1 = 2 * rndm.random()
            C2 = 2 * rndm.random()
            C3 = 2 * rndm.random()
            X1 = [0.0 for i in range(dim)]
            X2 = [0.0 for i in range(dim)]
            X3 = [0.0 for i in range(dim)]
            X_new = [0.0 for i in range(dim)]
            for j in range(dim):
                X1[j] = alpha.position[j] - A1 * abs(C1 * alpha.position[j] - population[i].position[j])
                X2[j] = beta.position[j] - A2 * abs(C2 * beta.position[j] - population[i].position[j])
                X3[j] = gamma.position[j] - A3 * abs(C3 * gamma.position[j] - population[i].position[j])
                X_new[j] += (X1[j] + X2[j] + X3[j])
            for j in range(dim):
                X_new[j] /= 3
            fitness_new = fitness(X_new)
            if fitness_new < population[i].fitness:
                population[i].position = X_new
                population[i].fitness = fitness_new

        population = sorted(population, key=lambda temp: temp.fitness)
        alpha = copy.copy(population[0])
        beta = copy.copy(population[1])
        gamma = copy.copy(population[2])
        iter += 1

    return alpha.position


def plot_rastrigin(gwo_solutions=None, names=None):
    X = np.linspace(-5.12, 5.12, 100)
    Y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(X, Y)

    Z = (X ** 2 - 10 * np.cos(2 * np.pi * X)) + \
        (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap=cm.coolwarm, linewidth=0.08,
                    antialiased=True)
    # plot the gwo solution
    # rotate the graph
    ax.view_init(90, 0)
    if gwo_solutions is not None:
        for gwo_solution in gwo_solutions:
            ax.scatter(gwo_solution[0], gwo_solution[1], gwo_solution[2], marker='*', color='red', s=100)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    if names is None:
        plt.savefig('outputs/rastrigin.pdf')
    else:
        plt.savefig(f'outputs/{names}.pdf')
    plt.show()


# plot the contour of the rastrigin function
def plot_rastrigin_contour(gwo_solutions=None, names=None):
    X = np.linspace(-5.12, 5.12, 100)
    Y = np.linspace(-5.12, 5.12, 100)
    X, Y = np.meshgrid(X, Y)

    Z = (X ** 2 - 10 * np.cos(2 * np.pi * X)) + \
        (Y ** 2 - 10 * np.cos(2 * np.pi * Y)) + 20
    plt.contour(X, Y, Z, 50, alpha=1.0, cmap='jet')
    # plot the gwo solution on top of the contour plot
    if gwo_solutions is not None:
        for gwo_solution in gwo_solutions:
            plt.plot(gwo_solution[0], gwo_solution[1], marker='*', color='red', markersize=8)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    if names is None:
        plt.savefig('outputs/rastrigin_contour.pdf')
    else:
        plt.savefig(f'outputs/rastrigin_contour_{names}.pdf')
    plt.show()


# create script to run gwo on different population sizes


def run_gwo_different_sizes():
    print("\nBegin grey wolf optimization on rastrigin function\n")
    dim = 5
    fitness = fitness_rastrigin

    print("Goal is to minimize Rastrigin's function in " + str(dim) + " variables")
    print("Function has known min = 0.0 at (", end="")
    for i in range(dim - 1):
        print("0, ", end="")
    print("0)")
    populations = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    computed_solutions = []
    computed_fitness = []
    for pop in populations:
        max_iter = 120

        print("Setting population = " + str(pop))
        print("Setting max_iter    = " + str(max_iter))
        print("\nStarting GWO algorithm\n")

        best_position = gwo(fitness, max_iter, pop, dim, -5.12, 5.12)

        print("\nGWO completed\n")
        print("\nBest solution found:")
        computed_solutions.append(best_position)
        print(["%.6f" % best_position[k] for k in range(dim)])
        err = fitness(best_position)
        computed_fitness.append(err)
        print("fitness of best solution = %.6f" % err)

        print("\nEnd GWO for rastrigin\n")

    # save population size, fitness and solutions to a csv file
    with open('outputs/gwo_rastrigin.csv', 'w') as f:
        f.write("population,fitness,solution\n")
        for i in range(len(populations)):
            f.write(str(populations[i]) + "," + str(computed_fitness[i]) + "," + str(computed_solutions[i]) + "\n")

    # make a plot of the fitness vs population size
    plt.plot(populations, computed_fitness)
    plt.xticks(populations)
    plt.yticks(np.arange(-3, 10, 1))
    plt.xlabel("Population Size")
    plt.ylabel("Fitness")
    plt.title("Fitness vs Population Size")
    plt.savefig('gwo_rastrigin_fitness.pdf')
    plt.show()

    plot_rastrigin(computed_solutions)
    plot_rastrigin_contour(computed_solutions)


def run_gwo(dim, populations=[15, 20, 25, 30, 35, 40, 45, 50, 55, 60], max_iter=120):
    fitness = fitness_rastrigin
    print("Goal is to minimize Rastrigin's function in " + str(dim) + " variables")
    print("Function has known min = 0.0 at (", end="")
    for i in range(dim - 1):
        print("0, ", end="")
    print("0)")
    computed_solutions = []
    computed_fitness = []
    for pop in populations:
        print("Setting population = " + str(pop))
        print("Setting max_iter    = " + str(max_iter))
        print("\nStarting GWO algorithm\n")

        best_position = gwo(fitness, max_iter, pop, dim, -5.12, 5.12)

        print("\nGWO completed\n")
        print("\nBest solution found:")
        computed_solutions.append(best_position)
        print(["%.6f" % best_position[k] for k in range(dim)])
        err = fitness(best_position)
        computed_fitness.append(err)
        print("fitness of best solution = %.6f" % err)

        print("\nEnd GWO for rastrigin\n")

    # pretty print the results
    print("Population size\tFitness\tSolution")
    for i in range(len(populations)):
        if math.isclose(computed_fitness[i], 0.0, abs_tol=1e-10):
            add_gwo_solutions_to_csv(dim, populations[i], computed_fitness[i], computed_solutions[i])

        print(str(populations[i]) + "\t" + str(computed_fitness[i]) + "\t" + str(computed_solutions[i]))


def add_gwo_solutions_to_csv(dim, population, computed_fitness, computed_solution):
    with open('outputs/gwo_rastrigin_optimal_sols_2.csv', 'a') as f:
        f.write(str(dim) + "," + str(population) + "," + str(computed_fitness) + "," + str(computed_solution) + "\n")


def run_gwo_different_dimensions():
    dims = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for dim in dims:
        print("Running GWO on dim = " + str(dim))
        run_gwo(dim)
        print("\n\n")


def main():
    run_gwo_different_dimensions()


if __name__ == "__main__":
    main()
