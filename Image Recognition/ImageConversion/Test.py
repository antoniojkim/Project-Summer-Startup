import numpy as np
import time
import ImageConversion as ic

input = []

for i in range(0, 1281):
    for j in range (0, 720):
        input.append((np.random.randint(0, 257), np.random.randint(0, 257), np.random.randint(0, 257)))

print("Input Generated")

start = int(round(time.time() * 1000))

converted = ic.convert_image(input)

end = int(round(time.time() * 1000))

print("Data Processed.  Took", (end-start), "Milliseconds to process data.")

for x,y in zip(input, converted):
    print(x, " ", y)