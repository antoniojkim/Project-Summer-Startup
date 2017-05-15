import numpy as np


combinations = np.load("./Combinations.npy")

def convert_image(input):
    length = len(input)
    if isinstance(input, list) and length > 0 and isinstance(input[0], tuple):
        def formula(i):
            return combinations[i[0]][i[1]][i[2]]
        return map(formula, input)
