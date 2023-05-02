import numpy as np
import matplotlib.pyplot as plt
import time


class HarryPlotter:

    @staticmethod
    def plot_rotational_vectors_zscore(rotational_vectors, sc=None):
        z_scores = np.abs((rotational_vectors - np.mean(rotational_vectors)) / np.std(rotational_vectors))
        red_green = False
        if red_green:

            if sc is None:
                fig, ax = plt.subplots()
                ax.set_xlabel('Index')
                ax.set_ylabel('Z-score')
                ax.set_title('Z-scores of Rotational Vectors')
                sc = ax.scatter(
                    x=range(len(rotational_vectors)),
                    y=np.zeros(len(rotational_vectors)),
                    c='blue'
                )

            is_outlier = np.all(z_scores < 2, axis=1)
            sc.set_offsets(np.column_stack((range(len(rotational_vectors)), z_scores[:, 0])))
            sc.set_color(['red' if o else 'blue' for o in is_outlier])

            plt.pause(1)
            plt.draw()
            time.sleep(2)
            plt.close()
            return sc

        else:

            plt.plot(z_scores)
            plt.xlabel('Index')
            plt.ylabel('Z-score')
            plt.title('Z-scores of Rotational Vectors')
            plt.show()

    @staticmethod
    def plot_distances(distance_dict, use_box=False):

        if use_box:
            distances = [distance_dict[size] for size in distance_dict]
            plt.boxplot(distances)
        else:
            for size, distances in distance_dict.items():
                plt.scatter([size] * len(distances), distances, color='blue', alpha=0.5)

        # Add labels and title
        plt.xlabel('Sample size')
        plt.ylabel('Euclidean distance')
        plt.title('Euclidean distance vs. sample size')

        # Show the plot
        plt.show()

    @staticmethod
    def plot_translation_vectors(translations):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x_vals = [t[0] for t in translations]
        y_vals = [t[1] for t in translations]
        z_vals = [t[2] for t in translations]

        ax.scatter(x_vals, y_vals, z_vals)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_xlim3d(-1, 1)
        # ax.set_ylim3d(-1, 1)
        # ax.set_zlim3d(-1, 1)

        plt.show()

    @staticmethod
    def plot_translation_vector_categories(translations):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        cmap = plt.get_cmap('hsv')
        num_colors = len(translations)
        colors = [cmap(i / num_colors) for i in range(num_colors)]

        for i, (label, points) in enumerate(translations.items()):
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            z_vals = [p[2] for p in points]
            # ax.scatter(x_vals, y_vals, z_vals, c=colors[i % len(colors)], label=label)
            ax.scatter(x_vals, y_vals, z_vals, c=colors[i], label=label)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.legend()
        plt.show()

    @staticmethod
    def plot_stds(translation_stds, rotation_stds):
        x = np.arange(3)
        y = translation_stds
        plt.errorbar(x, np.zeros(3), y, fmt='o', color='blue', ecolor='lightblue', elinewidth=3, capsize=0)

        x = np.arange(4)
        y = rotation_stds
        plt.errorbar(x, np.zeros(4), y, fmt='o', color='red', ecolor='pink', elinewidth=3, capsize=0)

        plt.xlabel('Axis')
        plt.ylabel('Standard deviation')
        plt.title('Standard deviations of translations and rotations')

        plt.show()
