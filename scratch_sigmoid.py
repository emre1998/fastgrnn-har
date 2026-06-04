"""
A "get a feel for sigmoid" script. Run it and look at the outputs.
Usage: venv active, then:  python scratch_sigmoid.py
"""

import numpy as np
from fastgrnn_numpy import sigmoid

# 1) Three canonical values to memorize
print("--- 1) Three canonical values ---")
print("sigmoid(0)   =", sigmoid(0))
print("sigmoid(10)  =", sigmoid(10))
print("sigmoid(-10) =", sigmoid(-10))

# 2) Does it work element-wise on a vector?
print("\n--- 2) On a vector ---")
print(sigmoid(np.array([-5, -1, 0, 1, 5])))

# 3) Closer to our real setting: an (8,) vector
print("\n--- 3) (8,) vector, gate simulation ---")
np.random.seed(0)  # reproducibility
pre = np.random.randn(8) * 2
gate = sigmoid(pre)
print("pre :", pre)
print("gate:", gate)
print("all in (0, 1)?", np.all((gate > 0) & (gate < 1)))
