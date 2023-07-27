import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


plt.switch_backend('agg')


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
        sns.histplot(data=data_frame, x='Distance', hue='Category', multiple='stack', palette='bright')
        plt.xlabel('Distance')
        plt.ylabel('Count')
        plt.title('Histogram of Distances by Category')

        plt.show()

    @staticmethod
    def plot_prop(data_frame, x="Distance", sort="Category"):
        plt.figure()
        sns.kdeplot(data=data_frame, x=x, hue=sort, multiple="stack")

        plt.title(f'{x} by {sort}')
        plt.xlabel(x)
        plt.ylabel("Density")
        plt.show()

    @staticmethod
    def plot_line(data_frame, x="Distance", y="Category", title="Line Plot"):
        plt.figure()

        sns.lineplot(data=data_frame, x=x, y=y)

        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    @staticmethod
    def plot_distance_density(frame, cols=["Translation X", "Translation Y", "Translation Z"]):
        data = frame[cols].abs()

        sns.kdeplot(data=data, fill=True)

        plt.xlabel("Distance")
        plt.ylabel("Density")
        plt.title("Distance Distribution")
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

    @staticmethod
    def plot_scatter_image_count(error_list, im_shape):
        fx = []
        fy = []
        cx = []
        cy = []
        for instance in error_list:
            fx.append(instance[0][0])
            fy.append(instance[1][1])
            cx.append(instance[0][2])
            cy.append(instance[1][2])

        figure, axis = plt.subplots(2, 2)
        axis[0, 0].plot(fx, label='fx')
        axis[0, 0].title.set_text('fx')
        axis[0, 0].set_ylim([0, max(im_shape)])
        axis[0, 1].plot(fy, label='fy')
        axis[0, 1].title.set_text('fy')
        axis[0, 1].set_ylim([0, max(im_shape)])
        axis[1, 0].plot(cx, label='cx')
        axis[1, 0].title.set_text('cx')
        axis[1, 0].set_ylim([0, max(im_shape)])
        axis[1, 1].plot(cy, label='cy')
        axis[1, 1].title.set_text('cy')
        axis[1, 1].set_ylim([0, min(im_shape[:2])])

        # plt.figure()

        # sns.lineplot(fx)
        # sns.lineplot(fy)
        # sns.lineplot(cx)
        # sns.lineplot(cy)

        plt.show()

    @staticmethod
    def plot_intrinsic_guess(error_list_1, error_list_2, im_shape):

        rpe_1 = []
        fx_1 = []
        fy_1 = []
        cx_1 = []
        cy_1 = []
        dist_1_0 = []
        dist_1_1 = []
        dist_1_2 = []
        dist_1_3 = []
        dist_1_4 = []
        for instance in error_list_1:
            dist = np.array(instance[2]).flatten()
            fx_1.append(instance[1][0][0])
            fy_1.append(instance[1][1][1])
            cx_1.append(instance[1][0][2])
            cy_1.append(instance[1][1][2])
            rpe_1.append(instance[0])

            dist_1_0.append(dist[0])
            dist_1_1.append(dist[1])
            dist_1_2.append(dist[2])
            dist_1_3.append(dist[3])
            dist_1_4.append(dist[4])
        rpe_2 = []
        fx_2 = []
        fy_2 = []
        cx_2 = []
        cy_2 = []
        dist_2_0 = []
        dist_2_1 = []
        dist_2_2 = []
        dist_2_3 = []
        dist_2_4 = []
        for instance in error_list_2:
            dist = np.array(instance[2]).flatten()
            fx_2.append(instance[1][0][0])
            fy_2.append(instance[1][1][1])
            cx_2.append(instance[1][0][2])
            cy_2.append(instance[1][1][2])
            rpe_2.append(instance[0])

            dist_2_0.append(dist[0])
            dist_2_1.append(dist[1])
            dist_2_2.append(dist[2])
            dist_2_3.append(dist[3])
            dist_2_4.append(dist[4])
        # plt.plot(rpe_1)
        # plt.plot(rpe_2)
        # figure, axis = plt.subplots(3, 3)
        # axis[0, 0].plot(fx_1, label='guess')
        # axis[0, 0].plot(fx_2, label='no guess')
        # axis[0, 0].title.set_text('fx')
        # axis[0, 0].legend()
        #
        # # axis[0, 0].set_ylim([0, max(im_shape)])
        # axis[0, 1].plot(fy_1, label='guess')
        # axis[0, 1].plot(fy_2, label='no guess')
        # axis[0, 1].title.set_text('fy')
        # # axis[0, 1].set_ylim([0, max(im_shape) + 100])
        # axis[1, 0].plot(cx_1, label='guess')
        # axis[1, 0].plot(cx_2, label='no guess')
        # axis[1, 0].title.set_text('cx')
        # # axis[1, 0].set_ylim([0, max(im_shape)])
        # axis[1, 1].plot(cy_1, label='guess')
        # axis[1, 1].plot(cy_2, label='no guess')
        # axis[1, 1].title.set_text('cy')
        # # axis[1, 1].set_ylim([0, max(im_shape)])
        #
        # axis[0, 2].plot(rpe_1, label='guess')
        # axis[0, 2].plot(rpe_2, label='no guess')
        # axis[0, 2].title.set_text('rpe')
        # # axis[0, 2].set_ylim([0, max(im_shape)])
        #
        # axis[1, 2].plot(dist_1_0, label='guess')
        # axis[2, 0].plot(dist_1_1, label='guess')
        # axis[2, 1].plot(dist_1_2, label='guess')
        # # axis[2, 2].plot(dist_1_3, label='guess')
        # axis[2, 2].plot(dist_1_4, label='guess')
        # axis[1, 2].plot(dist_2_0, label='no guess')
        # axis[2, 0].plot(dist_2_1, label='no guess')
        # axis[2, 1].plot(dist_2_2, label='no guess')
        # # axis[2, 2].plot(dist_2_3, label='no guess')
        # axis[2, 2].plot(dist_2_4, label='no guess')
        # axis[1, 2].title.set_text('distortion')
        # # axis[1, 1].set_ylim([0, max(im_shape)])
        figure, axis = plt.subplots()
        axis.plot(fx_1, label='Far Away')
        axis.plot(fx_2, label='Near')
        axis.title.set_text('fx')
        plt.xlabel('Image count')
        plt.ylabel('Pixels')
        plt.legend()
        plt.show()
        figure, axis = plt.subplots()
        axis.plot(fy_1, label='Far Away')
        axis.plot(fy_2, label='Near')
        axis.title.set_text('fy')
        plt.xlabel('Image count')
        plt.ylabel('Pixels')
        plt.legend()
        plt.show()
        figure, axis = plt.subplots()
        axis.plot(cx_1, label='Far Away')
        axis.plot(cx_2, label='Near')
        axis.title.set_text('cx')
        plt.xlabel('Image count')
        plt.ylabel('Pixels')
        plt.legend()
        plt.show()
        figure, axis = plt.subplots()
        axis.plot(cy_1, label='Far Away')
        axis.plot(cy_2, label='Near')
        axis.title.set_text('cy')
        plt.xlabel('Image count')
        plt.ylabel('Pixels')
        plt.legend()
        plt.show()
        figure, axis = plt.subplots()
        axis.plot(rpe_1, label='Far Away')
        axis.plot(rpe_2, label='Near')
        axis.title.set_text('Reprojection Error')
        plt.xlabel('Image count')
        plt.ylabel('Pixels')
        plt.legend()
        plt.show()
        figure, axis = plt.subplots()
        axis.plot(dist_1_0, label='Far Away')
        axis.plot(dist_2_0, label='Near')
        axis.title.set_text('k1')
        plt.xlabel('Image count')
        plt.ylabel('Pixels')
        plt.legend()
        plt.show()
        figure, axis = plt.subplots()
        axis.plot(dist_1_1, label='Far Away')
        axis.plot(dist_2_1, label='Near')
        axis.title.set_text('k2')
        plt.xlabel('Image count')
        plt.ylabel('Pixels')
        plt.legend()
        plt.show()
        figure, axis = plt.subplots()
        axis.plot(dist_1_2, label='Far Away')
        axis.plot(dist_2_2, label='Near')
        axis.title.set_text('p1')
        plt.xlabel('Image count')
        plt.ylabel('Pixels')
        plt.legend()
        plt.show()
        figure, axis = plt.subplots()
        axis.plot(dist_1_3, label='Far Away')
        axis.plot(dist_2_3, label='Near')
        axis.title.set_text('p2')
        plt.xlabel('Image count')
        plt.ylabel('Pixels')
        plt.legend()
        plt.show()
        figure, axis = plt.subplots()
        axis.plot(dist_1_4, label='Far Away')
        axis.plot(dist_2_4, label='Near')
        axis.title.set_text('k3')
        plt.xlabel('Image count')
        plt.ylabel('Pixels')
        plt.legend()

        # sns.lineplot(error_list_1)
        # sns.lineplot(error_list_2)
        # plt.legend('first', 'second')
        plt.show()
