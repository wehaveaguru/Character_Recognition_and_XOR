import numpy as np
import pandas as pd

df = pd.read_csv("emnist-byclass-train.csv", header=None, dtype=np.uint8, engine="pyarrow")
y = df.iloc[:, 0].values
mask = (y >= 1) & (y <= 35)
X = df.iloc[:, 1:].values[mask]
y = y[mask] - 1
del df

print(np.bincount(y))   # note the smallest count

per_class = 2000        # must be <= the smallest count above
rng = np.random.default_rng(42)
idx = np.concatenate([
    rng.choice(np.where(y == c)[0], per_class, replace=False)
    for c in range(35)
])
rng.shuffle(idx)

np.save("X_train.npy", X[idx])
np.save("y_train.npy", y[idx])

print(X.shape, y.shape)         # (70000, 784) (70000,)
print(np.bincount(y))           # 35 values, all 2000
print(X.min(), X.max())         # 0 255 (or 0.0 1.0 after scaling)
print(np.bincount(y).min())
print(np.bincount(y[idx]))   # should be 35 values, all 2000