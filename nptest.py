
import numpy as np

with open ("matcher.npy", "rb") as f:
    d1 = np.load(f, allow_pickle = True)
with open ("songs.npy", "rb") as f:
    d2 = np.load(f, allow_pickle = True)

print(d1)
print(d2)



