import numpy as np
from random import randint
table ={2**i:i for i in range(1,16)}
table[0]=0
def make_input(grid):
    g0 = grid
    print(table)
    r = np.zeros(shape=(16, 4, 4), dtype=float)
    for i in range(4):
        for j in range(4):
            v = g0[i, j]
            print(v)
            r[table[v],i, j]=1
    return r



testa = np.array(
    [[64, 32, 2 ** 14, 16],
    [16, 2, 4, 8],
    [1024, 2 ** 15, 32., 2 ** 13],
    [4096, 8, 16, 0]], dtype = int)
#testa = int(testa)
print(make_input(testa))