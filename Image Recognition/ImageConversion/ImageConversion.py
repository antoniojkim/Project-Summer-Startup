import numpy as np
import os.path

combinations = []
if (os.path.exists("./Combinations.npy")):
    combinations = np.load("./Combinations.npy")
else:
    colour_range = range(0, 257)
    logy = np.log(1367)
    logz = np.log(16001)

    for x in colour_range:
        d1 = []
        for y in colour_range:
            d2 = []
            for z in colour_range:
                d2.append(np.log1p(x) + np.log1p(y) / logy + np.log1p(z) / logz)
            d1.append(d2)
        combinations.append(d1)

    np.save("./Combinations.npy", combinations)

def convert_image(input):
    length = len(input)
    if isinstance(input, list) and length > 0 and isinstance(input[0], tuple):
        def formula(i):
            return combinations[i[0]][i[1]][i[2]]
        return map(formula, input)
