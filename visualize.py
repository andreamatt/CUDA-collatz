import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BATCH_SIZE = 1024
stride = 1000
n_values = 100000

point_size = 1
dpi = 1000

AVG = np.fromfile("output/avg.bin", dtype=np.uint16)
MIN = np.fromfile("output/min.bin", dtype=np.uint16)
MAX = np.fromfile("output/max.bin", dtype=np.uint16)

print(f"Total points: {len(AVG)}")

# reduce using stride
AVG = AVG[::stride]
MIN = MIN[::stride]
MAX = MAX[::stride]

# reduce using n_values
# AVG = AVG[:n_values]
# MIN = MIN[:n_values]
# MAX = MAX[:n_values]

X = np.arange(0, len(AVG)) * BATCH_SIZE
data = pd.DataFrame({"Average": AVG, "Min": MIN, "Max": MAX, "N": X})

# plot avg using seaborn
sns.scatterplot(data=data, x="N", y="Average", s=point_size, color="blue", alpha=0.5)
plt.savefig("images/avg.png", dpi=dpi)
plt.clf()

# plot min
sns.scatterplot(data=data, x="N", y="Min", s=point_size, color="red", alpha=0.5)
plt.savefig("images/min.png", dpi=dpi)
plt.clf()

# plot max
sns.scatterplot(data=data, x="N", y="Max", s=point_size, color="green", alpha=0.5)
plt.savefig("images/max.png", dpi=dpi)
plt.clf()

# plot all three
all_df = pd.DataFrame(
    {
        "N": np.concatenate((X, X, X)),
        "Steps": np.concatenate((AVG, MIN, MAX)),
        "Metric": np.array(["Average"] * len(AVG) + ["Min"] * len(AVG) + ["Max"] * len(AVG)),
    }
)
sns.scatterplot(data=all_df, x="N", y="Steps", hue="Metric", s=point_size, palette=["blue", "red", "green"], alpha=0.5)
plt.savefig("images/all.png", dpi=dpi)
