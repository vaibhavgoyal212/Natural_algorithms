
def add(x, y): return x + y


def sub(x, y): return x - y


def mul(x, y): return x * y


FUNCTIONS = [add, sub, mul]
TERMINALS = ["x", *range(1, 10)]
DOMAIN = (1, 100, 1)


# implement target function for bell sequence
def target_func(x):
    x=int(x)
    s = [[0 for _ in range(x + 1)] for _ in range(x + 1)]
    for i in range(x + 1):
        for j in range(x + 1):
            if j > i:
                continue
            elif (i == j):
                s[i][j] = 1
            elif (i == 0 or j == 0):
                s[i][j] = 0
            else:
                s[i][j] = j * s[i - 1][j] + s[i - 1][j - 1]
    ans = 0
    for i in range(0, x + 1):
        ans += s[x][i]
    return ans


def get_name():
    return "bell"