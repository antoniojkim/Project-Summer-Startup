import numpy as np

logy = np.log(1367)
logz = np.log(16001)

def convert_image(input):
    length = len(input)
    if isinstance(input, list) and length > 0 and isinstance(input[0], tuple):
        def formula(i):
            return np.log1p(i[0])+np.log1p(i[1])/logy+np.log1p(i[2])/logz
        return map(formula, input)
