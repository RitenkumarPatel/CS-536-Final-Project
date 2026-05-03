import numpy as np
from data import Data, LinearEvent, CompositeEvent

N = 10

e1 = LinearEvent(2.0, 0.0)  # 2x
e2 = LinearEvent(-1.0, 3.0)  # -x + 3
e3 = LinearEvent(0.5, 1.0)  # 0.5x + 1

# c1 = e1 + e2, c2 = e1 + e3
c1 = CompositeEvent([e1, e2], np.array([1.0, 1.0]))
c2 = CompositeEvent([e1, e3], np.array([1.0, 1.0]))

data = Data(N)
matrix = data.get([c1, c2])

print("shape:", matrix.shape)  # expect (2, 100)

# verify c1: e1 + e2 = 2x + (-x+3) = x + 3, mean ~ 3
e1_vals = data.get_single(e1)
e2_vals = data.get_single(e2)
e3_vals = data.get_single(e3)

assert np.allclose(matrix[0], e1_vals + e2_vals), "c1 mismatch"
assert np.allclose(matrix[1], e1_vals + e3_vals), "c2 mismatch"

print("c1 mean:", matrix[0].mean(), " (expect ~3)")
print("c2 mean:", matrix[1].mean(), " (expect ~1)")
print("all assertions passed")

print(data.get([e1, e2, e3, c1, c2]))
