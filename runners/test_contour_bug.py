import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Test how ax.contour works with origin="upper"
img = np.zeros((100, 100))
img[20:40, 20:40] = 1 # Top-left square

fig, ax = plt.subplots()
ax.imshow(img, origin="upper")
ax.contour(img, levels=[0.5], origin="upper")
fig.savefig("runners/outputs/test_contour_upper.png")
plt.close(fig)

fig, ax = plt.subplots()
ax.imshow(img, origin="upper")
ax.contour(img, levels=[0.5]) # without origin
fig.savefig("runners/outputs/test_contour_default.png")
plt.close(fig)

# Check the coordinate of the contour line:
cs_upper = ax.contour(img, levels=[0.5], origin="upper")
for path in cs_upper.get_paths():
    v = path.vertices
    print("Contour with origin='upper' vertices Y range:", v[:, 1].min(), v[:, 1].max())

cs_default = ax.contour(img, levels=[0.5])
for path in cs_default.get_paths():
    v = path.vertices
    print("Contour default vertices Y range:", v[:, 1].min(), v[:, 1].max())
