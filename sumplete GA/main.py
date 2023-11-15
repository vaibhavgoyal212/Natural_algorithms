from sumplete import genetic_algorithm, create_problem
import numpy as np
import matplotlib.pyplot as plt


def plot(fitness_function: str):
    DIMS = []
    average_num_generations = []
    with open(f"outputs/{fitness_function}_results.csv", "r") as f:
        for line in f.readlines():
            print(line)
            DIMS.append(int(line.split(",")[0]))
            average_num_generations.append(float(line.split(",")[1][:-1]))
    print(average_num_generations)
    plt.figure(figsize=(10, 6))
    plt.plot(DIMS, average_num_generations)
    # write number of generations on each point if less than 5001 else write no solution found
    for i in range(len(DIMS)):
        if average_num_generations[i] < 5000:
            plt.text(DIMS[i], average_num_generations[i], str(average_num_generations[i]))
        else:
            plt.text(DIMS[i], average_num_generations[i]-100, "No solution" + "\n" + "found")
    # add markers to points
    plt.scatter(DIMS, average_num_generations, marker="o", color="red")
    plt.xlabel("Dimensions")
    plt.ylabel("Average number of generations")
    plt.title(f"Average number of generations over dimensions for {fitness_function} fitness function")

    plt.savefig(f"outputs/average_num_generations_{fitness_function}.pdf")
    plt.show()


def main():
    DIMS = [3, 4, 5, 6]
    average_num_generations = []
    generations = 5000
    population_size = 300
    mutation_rate = 0.01
    xo_point_rate = 0.7
    fitness_function = "dynamic"

    for dim in DIMS:
        gens_over_repitions = []
        for i in range(5):
            problem = create_problem(dim)
            solution = genetic_algorithm(problem, generations, population_size, mutation_rate, xo_point_rate,
                                         fitness_function)
            gens_over_repitions.append(solution[1])
        average_num_generations.append(np.mean(gens_over_repitions))
    print(average_num_generations)

    with open(f"outputs/{fitness_function}_results.csv", "a") as f:
        for j in range(len(DIMS)):
            f.write(str(DIMS[j]) + "," + str(average_num_generations[j]) + "\n")

    # plot dimensions vs generations
    plot(fitness_function)


if __name__ == "__main__":
    main()
