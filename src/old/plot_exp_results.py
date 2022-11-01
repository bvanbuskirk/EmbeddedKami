import os

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

path = "/Users/Obsidian/Desktop/eecs106b/projects/MPCDynamicsKamigami/sim/data/loss"

robot0_data = []
robot0_names = []
robot2_data = []
robot2_names = []
ded_data = []
ded_names = []

for root, dirs, files in os.walk(path):
    for name in files:
        data = np.load(os.path.join(root, name))
        if "robot2_ded" in name:
            ded_data.append(data)
            ded_names.append(name)
        elif "robot0" in name:
            robot0_data.append(data)
            robot0_names.append(name)
        elif "robot2" in name:
            robot2_data.append(data)
            robot2_names.append(name)
        else:
            raise ValueError

print(len(robot0_data), len(robot2_data), len(ded_data))

robot0_data = np.array(robot0_data)
robot2_data = np.array(robot2_data)
ded_data = np.array(ded_data)

# data_names = ["Steps", "Total Perpendicular Loss", "Total Heading Loss"]
# model_names = ["JN", "JP", "JN", "SN", "SP"]
model_names = ["Joint\nNormal", "Joint\nPerturbed", "Joint\nNaive", "Single\nNormal", "Single\nPerturbed"]

new_robot0_data = robot0_data.copy()
new_robot2_data = robot2_data.copy()

for robot_names, robot_data, new_data in zip([robot0_names, robot2_names], [robot0_data, robot2_data], [new_robot0_data, new_robot2_data]):
    for i, name in enumerate(robot_names):
        if "joint2" in name:
            j = 0
        elif "joint0" in name:
            j = 1
        elif "naive" in name:
            j = 2
        elif "single2" in name:
            j = 3
        elif "single0" in name:
            j = 4
        else:
            raise ValueError
        new_data[j] = robot_data[i]

fig, ax = plt.subplots(2, 3)
fig.set_size_inches(15, 8.5)
width = 0.35
x = np.arange(len(model_names))

r0 = new_robot0_data
r2 = new_robot2_data

for j, robot in enumerate([r2, r0]):
    color = "blue" if j == 0 else "orange"
    for i in range(3):
        if j == 0:
            label = "Unperturbed Robot" if i == 0 else None
        else:
            label = "Perturbed Robot" if i == 0 else None
        
        if i == 0:
            ylabel = "Total # of Steps"
        elif i == 1:
            ylabel = "Total Perpendicular Cost"
        else:
            ylabel = "Total Heading Cost"
        if i == 0:
            bar = ax[j, i].bar(model_names, robot[:, i, 0], yerr=robot[:, i, 1], color=color, label=label, error_kw=dict(lw=2, capsize=5, capthick=2))
        else:
            bar = ax[j, i].bar(model_names, robot[:, i, 0]/robot[:, 0, 0], yerr=robot[:, i, 1]/robot[:, 0, 0], color=color, label=label, error_kw=dict(lw=2, capsize=5, capthick=2))
        ax[j, i].set_xticklabels(model_names, rotation=45)
        if i == 0:
            ax[j, i].plot(x, robot[:, i, 2], 'r.', label="Minimum Value" if i + j == 0 else None, markersize=15)
            ax[j, i].plot(x, robot[:, i, 3], 'g.', label="Maximum Value" if i + j == 0 else None, markersize=15)
        else:
            ax[j, i].plot(x, robot[:, i, 2]/robot[:, 0, 0], 'r.', label="Minimum Value" if i + j == 0 else None, markersize=15)
            ax[j, i].plot(x, robot[:, i, 3]/robot[:, 0, 0], 'g.', label="Maximum Value" if i + j == 0 else None, markersize=15)
        ax[j, i].grid(axis="y", linestyle="dotted", linewidth=0.5)
        ax[j, i].set_ylabel(ylabel)

# fig.legend(bbox_to_anchor=(0.5, 0.5))

# p2 = ax[0].bar(x + width / 2, r2[:, 0, 0], width, yerr=r2[:, 0, 1], label="Unperturbed Robot")

# ax[0, 0].plot([], [], label="J")

# ax[0].bar_label(p2, padding=3)

fig.legend(loc="right")
plt.subplots_adjust(left=0.05, right=0.85, top=0.98, bottom=0.1, hspace=0.3)
plt.show()

