def add(x, y):
    return x + y


def sub(x, y):
    return x - y


def mul(x, y):
    return x * y


FUNCTIONS = [add, sub, mul]
TERMINALS = ["x", -2, -1, 0, 1, 2]
DOMAIN = (-100, 100, 2)


def target_func(x):
    return x * x + 2 * x + 1


def get_name():
    return "quadratic"