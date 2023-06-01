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
    def plot_histogram_by_category(data_frame):
        print(data_frame)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data_frame, x='Distance', hue='Category', multiple='stack')
        plt.xlabel('Distance')
        plt.ylabel('Count')
        plt.title('Histogram of Distances by Category')

        plt.show()

    @staticmethod
    def plot_prop(data_frame, density=True, x_axis=None):
        print(data_frame)
        plt.figure()
        if density:
            sns.kdeplot(data=data_frame, x=x_axis, hue="Category", multiple="stack")
            # sns.kdeplot(data=data_frame, x="Distance", hue="Category", multiple="fill")
            # sns.kdeplot(
            #     data=data_frame, x="Distance", hue="Category",
            #     fill=True, common_norm=False, palette="crest",
            #     alpha=.5, linewidth=0,
            # )

        else:
            sns.histplot(data_frame, x=x_axis, hue="Category", element="poly")

        plt.title(f'{x_axis} by Category')
        plt.xlabel(f'{x_axis}')
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
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    @staticmethod
    def plot_variance(dataframe):
        # Calculate the variance for each column
        variance = dataframe

        # Plot the variance using seaborn
        # sns.barplot(x=variance.index, y=variance.values)
        sns.barplot(data=variance.reset_index(), x='Category', y=variance.values)

        # Set labels and title
        plt.xlabel("Columns")
        plt.ylabel("Variance")
        plt.title("Variance of Columns")

        # Rotate x-axis labels if needed
        plt.xticks(rotation=90)

        # Display the plot
        plt.show()

    @staticmethod
    def stacked_histogram(dataframe):
        # Set the figure size
        plt.figure(figsize=(10, 6))

        # Set the style
        sns.set(style="whitegrid")

        # Plot the stacked histogram
        sns.histplot(data=dataframe.reset_index(), x="Category", multiple="stack", stat="count", binwidth=1)

        # Set the labels and title
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.title("Stacked Histogram")

        # Show the plot
        plt.show()
