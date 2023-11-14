import numpy as np
import itertools
import math
from Natural_algorithms.gwo.gwo import plot_rastrigin_contour
import os
import matplotlib.pyplot as plt


def fitness_rastrigin(position, dim):
    fitness_value = 0.0
    for i in range(dim):
        xi = position[i]
        fitness_value += ((xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10)
    return fitness_value


class Particle:

    def __init__(self, dim, minx, maxx):
        self.position = np.random.uniform(low=minx, high=maxx, size=dim)
        self.velocity = np.random.uniform(low=-0.1, high=0.1, size=dim)
        self.best_particle_pos = self.position
        self.dim = dim

        self.fitness = fitness_rastrigin(self.position, dim)
        self.best_particle_fitness = self.fitness  # we couldd start with very large number here,
        # but the actual value is better in case we are lucky

    def setPos(self, pos):
        self.position = pos
        self.fitness = fitness_rastrigin(self.position, self.dim)
        if self.fitness < self.best_particle_fitness:
            self.best_particle_fitness = self.fitness
            self.best_particle_pos = pos

    def updateVel(self, inertia, a1, a2, best_self_pos, best_swarm_pos):
        cur_vel = self.velocity
        r1 = np.random.uniform(low=0, high=1, size=self.dim)
        r2 = np.random.uniform(low=0, high=1, size=self.dim)
        a1r1 = np.multiply(a1, r1)
        a2r2 = np.multiply(a2, r2)
        best_self_dif = np.subtract(best_self_pos, self.position)
        best_swarm_dif = np.subtract(best_swarm_pos, self.position)
        new_vel = inertia * cur_vel + np.multiply(a1r1, best_self_dif) + np.multiply(a2r2, best_swarm_dif)
        self.velocity = new_vel
        return new_vel


class PSO:

    def __init__(self, w, a1, a2, dim, population_size, time_steps, search_range):

        self.w = w
        self.a1 = a1
        self.a2 = a2
        self.dim = dim

        self.swarm = [Particle(dim, -search_range, search_range) for i in range(population_size)]
        self.time_steps = time_steps
        print('init')

        self.best_swarm_pos = np.random.uniform(low=-500, high=500, size=dim)
        self.best_swarm_fitness = 1e100

    def run(self, output=True):
        for t in range(self.time_steps):
            for p in range(len(self.swarm)):
                particle = self.swarm[p]

                new_position = particle.position + particle.updateVel(self.w, self.a1, self.a2,
                                                                      particle.best_particle_pos, self.best_swarm_pos)

                if new_position @ new_position > 1.0e+18:  # The search will be terminated if the distance
                    # of any particle from center is too large
                    print('Time:', t, 'Best Pos:', self.best_swarm_pos, 'Best Fit:', self.best_swarm_fitness)
                    raise SystemExit('Most likely divergent: Decrease parameter values')

                self.swarm[p].setPos(new_position)

                new_fitness = fitness_rastrigin(new_position, self.dim)

                if new_fitness < self.best_swarm_fitness:
                    self.best_swarm_fitness = new_fitness
                    self.best_swarm_pos = new_position

            if t % 100 == 0:
                print("Time: %6d,  Best Fitness: %14.6f,  Best Pos: %9.4f,%9.4f" % (
                    t, self.best_swarm_fitness, self.best_swarm_pos[0], self.best_swarm_pos[1]), end=" ")
                if self.dim > 2:
                    print('...')
                else:
                    print('')
        if output:
            print("Final:")
            print("Best Swarm Position:", self.best_swarm_pos[:2])
            print("Best Swarm Fitness:", self.best_swarm_fitness)
            return self.best_swarm_pos, self.best_swarm_fitness


def plot_fitness(populations, solutions):
    plt.plot(populations, solutions)
    plt.xticks(populations)
    plt.yticks(np.arange(-3, 10, 1))
    plt.xlabel("Population Size")
    plt.ylabel("Fitness")
    plt.title("Fitness vs Population Size")
    plt.savefig('outputs/pso_rastrigin_fitness2.pdf')
    plt.show()


def save_data(populations, solutions, fitness):
    # plot results of bat algorithm on rastrigin function contour
    plot_rastrigin_contour(solutions, "2")
    plot_fitness(populations, fitness)
    # save population, solutions and fitness in csv file in an output folder
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    np.savetxt("outputs/pso_rastrigin_2.csv", np.column_stack((populations, solutions, fitness)),
               delimiter=",", fmt='%s')


def pso_different_pops():
    # parameters
    dim = 3
    max_iteration = 100
    populations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    computed_solutions = []
    computed_fitness = []
    # run bat algorithm
    for population in populations:
        print("Population:", population)
        results = PSO(dim=dim, w=0.7, a1=2.02, a2=2.02, population_size=population, time_steps=max_iteration,
                      search_range=5.12).run()
        computed_solutions.append(results[0])
        computed_fitness.append(results[1])

    # save data
    save_data(populations, computed_solutions, computed_fitness)


def pso_different_dimensions():
    dims = [15]
    result_pops = []
    for dim in dims:
        print("Dimension:", dim)
        max_iteration = 700
        i = 1
        population = 60
        while True:
            print("Population:", population)
            results = PSO(dim=dim, w=0.7, a1=2.02, a2=2.02, population_size=population, time_steps=max_iteration,
                          search_range=5.12).run()
            if math.isclose(results[1], 0.0, abs_tol=1e-3):
                # append dimension, population, fitness and solution to result_data
                result_pops.append(population)
                break
            population += 40
            i += 1
            if i > 10:
                max_iteration += 100
        continue
    # save data with columns: dimension, population, fitness, solution to existing csv file
    with open('outputs/pso_rastrigin_dimensions.csv', 'a') as f:
        for i in range(len(result_pops)):
            f.write(str(dims[i]) + "," + str(result_pops[i]) + "\n")


def graph_pso_different_dimensions():
    # read data from csv file
    data = np.genfromtxt('outputs/pso_rastrigin_dimensions.csv', delimiter=',')
    # plot results
    plt.plot(data[:, 0], data[:, 1])
    plt.xticks(data[:, 0])
    plt.yticks(np.arange(0, 1500, 100))
    plt.xlabel("Dimensions")
    plt.ylabel("Population")
    plt.title("Population vs Dimensions")
    plt.savefig('outputs/pso_rastrigin_dimensions.pdf')
    plt.show()


def main():
    graph_pso_different_dimensions()


if __name__ == "__main__":
    main()
