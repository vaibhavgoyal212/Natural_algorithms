from gp_architecture import gp
from bell_skeleton import TERMINALS, FUNCTIONS, DOMAIN, target_func, get_name


if __name__ == "__main__":
    POP_SIZE = 50  # population size
    MIN_DEPTH = 4  # minimal initial random tree depth
    MAX_DEPTH = 9  # maximal initial random tree depth
    GENERATIONS = 300  # maximal number of generations to run evolution
    TOURNAMENT_SIZE = 5  # size of tournament for tournament selection
    XO_RATE = 0.7  # crossover rate
    PROB_MUTATION = 0.05  # per-node mutation probability
    name = get_name()
    gp(FUNCTIONS, TERMINALS, target_func, POP_SIZE, MIN_DEPTH, MAX_DEPTH, GENERATIONS, TOURNAMENT_SIZE, XO_RATE,PROB_MUTATION, DOMAIN, name)

