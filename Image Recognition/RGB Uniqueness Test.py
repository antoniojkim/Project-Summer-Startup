import numpy as np

colour_range = range(0, 257)
logy = np.log(1367)
logz = np.log(16001)

combinations = []

for x in colour_range:
    for y in colour_range:
        for z in colour_range:
            combinations.append(np.log1p(x)+np.log1p(y)/logy+np.log1p(z)/logz)

print("traversed")
combinations.sort()
print("sorted")

count_duplicates = 0
for i in range(1, len(combinations)):
    if combinations[i] == combinations[i-1]:
       count_duplicates += 1

print("Found ", count_duplicates, " Duplicates")

print("complete")