import numpy as np

colour_range = range(0, 257)
logy = np.log(5)
logz = np.log(7)

combinations = []

for x in colour_range:
    for y in colour_range:
        for z in colour_range:
            combinations.append(np.log1p(x)+np.log1p(y)/logy+np.log1p(z)/logz)

print("traversed")
np.sort(combinations)
print("sorted")

for i in range(1, len(combinations)):
    if combinations[i] == combinations[i-1]:
        print("Duplicate Found:  ", combinations[i])

print("complete")