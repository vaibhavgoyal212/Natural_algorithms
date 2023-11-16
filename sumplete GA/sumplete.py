# make a sumplete solver with genetic algorithm
from copy import deepcopy

import numpy as np
import random
import math
import matplotlib.pyplot as plt


# create random problem with kxk matrix and max value of 100
def create_problem(k: int):
    problem = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            problem[i][j] = random.randint(1, 100)

    # create a random solution
    solution = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            solution[i][j] = random.choices([0, 1], weights=[1 / 3, 2 / 3])[0]

    row_sums = []
    column_sums = []
    for i in range(k):
        row_curr = 0
        col_curr = 0
        for j in range(k):
            row_curr += int(problem[i][j] * solution[i][j])
            col_curr += int(problem[j][i] * solution[j][i])
        row_sums.append(row_curr)
        column_sums.append(col_curr)
    return problem, row_sums, column_sums


def pretty_print(problem, row_sums, col_sums):
    for i in range(len(problem)):
        for j in range(len(problem[0])):
            print(f"{problem[i][j]:2}", end=" ")
        print("|", row_sums[i])

    print("-" * (len(problem[0]) * 3 + len(str(row_sums[0])) + 1))

    for j in range(len(col_sums)):
        print(f"{col_sums[j]:2}", end=" ")
    print()


def fitness_binary(problem, row_sums, col_sums, solution):
    for i in range(len(problem)):
        row_curr = 0
        col_curr = 0
        for j in range(len(problem[0])):
            row_curr += int(problem[i][j] * solution[i][j])
            col_curr += int(problem[j][i] * solution[j][i])
        if row_curr != row_sums[i] or col_curr != col_sums[i]:
            return 0
    return 1


def fitness_dynamic(problem, row_sums, col_sums, solution):
    row_fitness = 0
    col_fitness = 0
    for i in range(len(problem)):
        row_curr = 0
        col_curr = 0
        for j in range(len(problem[0])):
            row_curr += int(problem[i][j] * solution[i][j])
            col_curr += int(problem[j][i] * solution[j][i])
        row_fitness += abs(row_curr - row_sums[i])
        col_fitness += abs(col_curr - col_sums[i])
        return 1 - ((row_fitness + col_fitness) / sum(row_sums + col_sums))


def mutation(solution, mutation_rate):
    for i in range(len(solution)):
        for j in range(len(solution[0])):
            if random.random() < mutation_rate:
                solution[i][j] = 1 - solution[i][j]
    return solution


def crossover(solution1, solution2, xo_point_rate):
    # single point crossover
    xo_point = int(len(solution1) * xo_point_rate)
    child1 = np.concatenate((solution1[:xo_point], solution2[xo_point:]))
    child2 = np.concatenate((solution2[:xo_point], solution1[xo_point:]))
    return child1, child2


def selection(problem, population,fitness_function):
    matrix, row_sums, col_sums = problem
    # tournament selection
    tournament_size = 5
    tournament = random.choices(population, k=tournament_size)
    if fitness_function == "binary":
        fitnesses = [fitness_binary(matrix, row_sums, col_sums, x) for x in tournament]
    else:
        fitnesses = [fitness_dynamic(matrix, row_sums, col_sums, x) for x in tournament]
    # get the best solution from the tournament using argmax
    return tournament[np.argmax(fitnesses)]


def get_new_population(problem, population, fitnesses, mutation_rate, xo_point_rate, fitness_function):
    # get top 10% of the population
    top_10 = np.argsort(fitnesses)[-int(len(population) * 0.1):]
    new_population = [population[i] for i in top_10]
    for i in range(len(population)):
        parent1 = selection(problem, population, fitness_function)
        parent2 = selection(problem,population, fitness_function)

        child1, child2 = crossover(parent1, parent2, xo_point_rate)
        child1 = mutation(child1, mutation_rate)
        child2 = mutation(child2, mutation_rate)
        new_population.append(child1)
        new_population.append(child2)
    return new_population[:len(population)]


def init_population(population_size, problem_size):
    population = []
    for i in range(population_size):
        solution = np.zeros((problem_size, problem_size))
        for j in range(problem_size):
            for k in range(problem_size):
                solution[j][k] = random.choices([0, 1], weights=[1 / 3, 2 / 3])[0]
        population.append(solution)
    return population


def plot_fitness(filename, generations, generation_best_fitnesses):
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(1, generations + 1)), generation_best_fitnesses)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("Best fitness over generations")
    plt.grid()
    plt.savefig(f"outputs/{filename}.png")
    plt.savefig(f"outputs/pdfs/{filename}.pdf")
    plt.show()


def genetic_algorithm(problem, generations, population_size, mutation_rate, xo_point_rate, fitness_function, graphing=True):
    matrix, row_sums, col_sums = problem
    population = init_population(population_size, len(matrix))
    best_fitness_overall = 0
    generation_best_fitnesses = []
    best_generation = 0
    best_solution = []
    for gen in range(1, generations + 1):
        print("generation:", gen)
        if fitness_function == "binary":
            fitness_fn = lambda x: fitness_binary(matrix, row_sums, col_sums, x)
        elif fitness_function == "dynamic":
            fitness_fn = lambda x: fitness_dynamic(matrix, row_sums, col_sums, x)
        else:
            raise Exception("Invalid fitness function")

        fitnesses = [fitness_fn(i) for i in population]
        generation_best_fitnesses.append(max(fitnesses))
        print("best fitness calculated")
        if max(fitnesses) > best_fitness_overall:
            best_fitness_overall = max(fitnesses)
            best_generation = gen
            best_solution = deepcopy(population[fitnesses.index(max(fitnesses))])
            print("best solution copied")
            print("_" * 50 + "\n")
            print("problem:" + "\n")
            pretty_print(matrix, row_sums, col_sums)
            print("generation:", gen, "best fitness:", max(fitnesses), "\n", "best solution:", best_solution)

        if best_fitness_overall == 1:
            print("Solution found")
            print("problem:")
            pretty_print(matrix, row_sums, col_sums)
            print("generation:", gen)
            print("solution:", best_solution)
            if graphing:
                plot_fitness(f"fitness_{fitness_function}_{len(matrix)}_matrix", gen, generation_best_fitnesses)
            return best_solution, gen

        population = get_new_population(problem,population, fitnesses, mutation_rate, xo_point_rate, fitness_function)
        print("new population created")

    print("Solution not found", "\n")
    print("best fitness:", best_fitness_overall, "in generation:", best_generation, "\n")
    print("best solution:", best_solution)
    if graphing:
        plot_fitness("fitness_dynamic", generations, generation_best_fitnesses)
    return None, generations


