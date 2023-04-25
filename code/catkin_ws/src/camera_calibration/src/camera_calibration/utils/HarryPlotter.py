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
                plt.scatter([size]*len(distances), distances, color='blue', alpha=0.5)

        # Add labels and title
        plt.xlabel('Sample size')
        plt.ylabel('Euclidean distance')
        plt.title('Euclidean distance vs. sample size')

        # Show the plot
        plt.show()