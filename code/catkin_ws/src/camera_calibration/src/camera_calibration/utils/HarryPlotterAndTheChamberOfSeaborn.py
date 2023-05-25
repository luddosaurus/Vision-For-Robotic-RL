import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class HarryPlotter:
    #  Panda Frame ['Category', 'X', 'Y', 'Z']

    @staticmethod
    def plot_3d_scatter(df, title="3D Scatter Plot"):
        sns.set(style="whitegrid")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        categories = df['Category'].unique()
        colors = sns.color_palette("hls", len(categories))

        df[['Translation X', 'Translation Y', 'Translation Z']] *= 100

        for i, category in enumerate(categories):
            category_data = df[df['Category'] == category]
            x = category_data['Translation X']
            y = category_data['Translation Y']
            z = category_data['Translation Z']
            ax.scatter(x, y, z, color=colors[i], label=category)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-100, 100)

        ax.legend()
        plt.show()

    @staticmethod
    def plot_std_deviation(std_deviation_frame, target='Translation X'):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=std_deviation_frame.reset_index(), x='Category', y=target)
        plt.xlabel('Category')
        plt.ylabel('Standard Deviation')
        plt.title('Standard Deviation of Columns by Category')
        plt.show()

    @staticmethod
    def plot_histogram_by_category(data_frame, density=True):
        plt.figure()
        if density:
            sns.kdeplot(data=data_frame, x="Distance", hue="Category", multiple="stack")
            # sns.kdeplot(data=data_frame, x="Distance", hue="Category", multiple="fill")
            # sns.kdeplot(
            #     data=data_frame, x="Distance", hue="Category",
            #     fill=True, common_norm=False, palette="crest",
            #     alpha=.5, linewidth=0,
            # )

        else:
            sns.histplot(data_frame, x="Distance", hue="Category", element="poly")

        plt.title('Distances by Category')
        plt.xlabel('Distance')
        plt.ylabel('Count')
        plt.show()

    @staticmethod
    def plot_prop(data_frame, density=True):
        plt.figure()
        if density:
            sns.kdeplot(data=data_frame, x="Distance", hue="Category", multiple="stack")
            # sns.kdeplot(data=data_frame, x="Distance", hue="Category", multiple="fill")
            # sns.kdeplot(
            #     data=data_frame, x="Distance", hue="Category",
            #     fill=True, common_norm=False, palette="crest",
            #     alpha=.5, linewidth=0,
            # )

        else:
            sns.histplot(data_frame, x="Distance", hue="Category", element="poly")

        plt.title('Distances by Category')
        plt.xlabel('Distance')
        plt.ylabel('Count')
        plt.show()

    @staticmethod
    def plot_poses(dataframe):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        translations = dataframe[['Translation X', 'Translation Y', 'Translation Z']].values
        rotations = dataframe[['Rotation X', 'Rotation Y', 'Rotation Z', 'Rotation W']].values

        for i in range(len(translations)):
            translation = translations[i]
            rotation = rotations[i]

            # Plot translation point
            ax.scatter(translation[0], translation[1], translation[2], color='magenta')

            # Extract rotational matrix from quaternion
            rot_matrix = np.array([[1 - 2 * (rotation[1] ** 2 + rotation[2] ** 2),
                                    2 * (rotation[0] * rotation[1] - rotation[2] * rotation[3]),
                                    2 * (rotation[0] * rotation[2] + rotation[1] * rotation[3])],
                                   [2 * (rotation[0] * rotation[1] + rotation[2] * rotation[3]),
                                    1 - 2 * (rotation[0] ** 2 + rotation[2] ** 2),
                                    2 * (rotation[1] * rotation[2] - rotation[0] * rotation[3])],
                                   [2 * (rotation[0] * rotation[2] - rotation[1] * rotation[3]),
                                    2 * (rotation[1] * rotation[2] + rotation[0] * rotation[3]),
                                    1 - 2 * (rotation[0] ** 2 + rotation[1] ** 2)]])

            # Plot rotation axes
            axis_colors = ['red', 'green', 'blue']
            for j in range(3):
                axis_start = translation
                axis_end = translation + 0.1 * rot_matrix[:, j]  # Use rotational matrix for axis direction
                ax.plot([axis_start[0], axis_end[0]], [axis_start[1], axis_end[1]], [axis_start[2], axis_end[2]],
                        color=axis_colors[j])

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
