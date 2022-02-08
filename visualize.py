import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

point_size = 0.3
stride = 2**10
AVG = np.fromfile("output/avg.bin", dtype=np.uint16)[0:-1:stride]
MIN = np.fromfile("output/min.bin", dtype=np.uint16)[0:-1:stride]
MAX = np.fromfile("output/max.bin", dtype=np.uint16)[0:-1:stride]
X = np.arange(0, len(AVG) * stride, stride)
data = pd.DataFrame({"Average": AVG, "Min": MIN, "Max": MAX, "Batch": X})

# plot avg using seaborn
sns.scatterplot(data=data, x="Batch", y="Average", s=point_size)
plt.savefig("images/avg.png", dpi=300)
plt.clf()

# plot min
sns.scatterplot(data=data, x="Batch", y="Min", s=point_size)
plt.savefig("images/min.png", dpi=300)
plt.clf()

# plot max
sns.scatterplot(data=data, x="Batch", y="Max", s=point_size)
plt.savefig("images/max.png", dpi=300)
plt.clf()

# plot all three
all_df = pd.DataFrame(
    {
        "Batch": np.concatenate((X, X, X)),
        "Steps": np.concatenate((AVG, MIN, MAX)),
        "Metric": np.array(["Average"] * len(AVG) + ["Min"] * len(AVG) + ["Max"] * len(AVG)),
    }
)
sns.scatterplot(data=all_df, x="Batch", y="Steps", hue="Metric", s=point_size)
plt.savefig("images/all.png", dpi=300)
