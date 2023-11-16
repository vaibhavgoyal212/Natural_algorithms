from random import random, randint, seed
from statistics import mean
from copy import deepcopy
import matplotlib.pyplot as plt


def generate_dataset(lo, hi, step):  # generate 101 data points from target_func
    dataset = []
    for x in range(lo, hi, step):
        x /= 100
        dataset.append([x, TARGET_FUNC(x)])
    return dataset


class GPTree:
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def node_label(self):  # string label
        if self.data in FUNCTIONS:
            return self.data.__name__
        else:
            return str(self.data)

    def print_tree(self, prefix=""):  # textual printout
        print("%s%s" % (prefix, self.node_label()))
        if self.left:  self.left.print_tree(prefix + "   ")
        if self.right: self.right.print_tree(prefix + "   ")

    def compute_tree(self, x):
        if self.data in FUNCTIONS:
            return self.data(self.left.compute_tree(x), self.right.compute_tree(x))
        elif self.data == 'x':
            return x
        else:
            return self.data

    def random_tree(self, grow, max_depth, depth=0):  # create random tree using either grow or full method
        if depth < MIN_DEPTH or (depth < max_depth and not grow):
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]
        elif depth >= max_depth:
            self.data = TERMINALS[randint(0, len(TERMINALS) - 1)]
        else:  # intermediate depth, grow
            if random() > 0.5:
                self.data = TERMINALS[randint(0, len(TERMINALS) - 1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS) - 1)]
        if self.data in FUNCTIONS:
            self.left = GPTree()
            self.left.random_tree(grow, max_depth, depth=depth + 1)
            self.right = GPTree()
            self.right.random_tree(grow, max_depth, depth=depth + 1)

    def mutation(self):
        if random() < PROB_MUTATION:  # mutate at this node
            self.random_tree(grow=True, max_depth=2)
        elif self.left:
            self.left.mutation()
        elif self.right:
            self.right.mutation()

    def size(self):  # tree size in nodes
        if self.data in TERMINALS: return 1
        left = self.left.size() if self.left else 0
        right = self.right.size() if self.right else 0
        return 1 + left + right

    def build_subtree(self):  # count is list in order to pass "by reference"
        tree = GPTree()
        tree.data = self.data
        if self.left:
            tree.left = self.left.build_subtree()
        if self.right:
            tree.right = self.right.build_subtree()
        return tree

    def scan_tree(self, count, second):  # note: count is list, so it's passed "by reference"
        count[0] -= 1
        if count[0] <= 1:
            if not second:  # return subtree rooted here
                return self.build_subtree()
            else:  # glue subtree here
                self.data = second.data
                self.left = second.left
                self.right = second.right
        else:
            ret = None
            if self.left and count[0] > 1: ret = self.left.scan_tree(count, second)
            if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)
            return ret

    def crossover(self, other):  # xo 2 trees at random nodes
        if random() < XO_RATE:
            second = other.scan_tree([randint(1, other.size())], None)  # 2nd random subtree
            self.scan_tree([randint(1, self.size())], second)  # 2nd subtree "glued" inside 1st tree


def fitness(individual, dataset):  # inverse mean absolute error over dataset normalized to [0,1]
    return 1 / (1 + mean([abs(individual.compute_tree(ds[0]) - ds[1]) for ds in dataset]))


def selection(population, fitnesses):  # select one individual using tournament selection
    tournament = [randint(0, len(population) - 1) for i in range(TOURNAMENT_SIZE)]  # select tournament contenders
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]])


def selection_roulette(population, fitnesses):  # select one individual using roulette selection
    total_fitness = sum(fitnesses)
    pick = random() * total_fitness
    current = 0
    for i in range(len(population)):
        current += fitnesses[i]
        if current > pick:
            return deepcopy(population[i])


def init_population():
    # ramped half-and-half
    pop = []
    for md in range(3, MAX_DEPTH + 1):
        for i in range(int(POP_SIZE / 6)):
            t = GPTree()
            t.random_tree(grow=True, max_depth=md)  # grow
            pop.append(t)
        for i in range(int(POP_SIZE / 6)):
            t = GPTree()
            t.random_tree(grow=False, max_depth=md)  # full
            pop.append(t)
    return pop


def plot_fitnesses(generations, fitnesses, target_func):
    num_range = [i for i in range(0,generations,4)] if generations < len(fitnesses) else [i for i in range(len(fitnesses))]
    plt.plot(num_range, fitnesses)
    #set x ticks to every 4th generation
    plt.xticks(num_range)
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.title(f"fitness over generations for {target_func} sequence")
    plt.savefig(f"outputs/fitnesses_{target_func}.pdf")
    plt.show()


def gp(functions, terminals, target_func, pop_size, min_depth, max_depth, generations, tournament_size, xo_rate,
       mutation_rate, function_domain, name):
    global FUNCTIONS, TERMINALS, PROB_MUTATION, POP_SIZE, MIN_DEPTH, MAX_DEPTH, GENERATIONS, TOURNAMENT_SIZE, XO_RATE, TARGET_FUNC
    FUNCTIONS = functions
    TERMINALS = terminals
    PROB_MUTATION = mutation_rate
    POP_SIZE = pop_size
    MIN_DEPTH = min_depth
    MAX_DEPTH = max_depth
    GENERATIONS = generations
    TOURNAMENT_SIZE = tournament_size
    XO_RATE = xo_rate
    TARGET_FUNC = target_func
    lo, hi, step = function_domain
    dataset = generate_dataset(lo, hi, step)
    population = init_population()
    best_solution_run = None
    best_fitness = 0
    best_generation = 0
    best_fitnesses_all_gens = []
    fitnesses = [fitness(individual, dataset) for individual in population]
    for g in range(GENERATIONS):
        print("running generation", g)
        new_population = []
        for i in range(POP_SIZE):  # create offspring one pair at a time
            parent1 = selection_roulette(population, fitnesses)
            parent2 = selection_roulette(population, fitnesses)
            parent1.crossover(parent2)
            parent1.mutation()
            new_population.append(parent1)
        population = new_population
        fitnesses = [fitness(individual, dataset) for individual in population]
        if max(fitnesses) > best_fitness:
            best_fitness = max(fitnesses)
            best_generation = g
            best_solution_run = deepcopy(population[fitnesses.index(max(fitnesses))])
            print("________________________")
            print("gen:", g, ", best_fitness:", round(max(fitnesses), 3), ", best_solution:")
            best_solution_run.print_tree()
        best_fitnesses_all_gens.append(best_fitness)
        if best_fitness == 1.0: break
    print("best individual:")
    print("\n\n_________________________________________________\nEND OF RUN\nbest run attained at gen " + str(
        best_generation) + \
          " and has f=" + str(round(best_fitness, 3)))
    best_solution_run.print_tree()
    plot_fitnesses(generations, best_fitnesses_all_gens, name)