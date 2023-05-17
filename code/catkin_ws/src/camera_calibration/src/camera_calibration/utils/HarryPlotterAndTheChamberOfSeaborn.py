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
    def plot_histogram_by_category(data_frame):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data_frame, x='Distance', hue='Category', multiple='stack')
        plt.xlabel('Distance')
        plt.ylabel('Count')
        plt.title('Histogram of Distances by Category')
        plt.legend(title='Category')
        plt.show()
