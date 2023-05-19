import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

    # Set labels and title
    ax.set_xlabel('Translation X')
    ax.set_ylabel('Translation Y')
    ax.set_zlabel('Translation Z')
    ax.set_title(title)

    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)

    ax.legend()
    plt.show()


def create_sample_data(num_categories, translation_samples, std=True):
    data = []

    for i in range(num_categories):
        sample_category = f"Cat_{i + 1}"

        if std:
            for j in range(translation_samples):
                mean_translation = [0.5, 0.2, 0.4]
                mean_rotation = [0.0, 0.0, 0.0, 0.0]
                std_dev = 0.2

                translation = np.random.normal(mean_translation, std_dev, 3)
                rotation = np.random.normal(mean_rotation, std_dev, 4)
                data.append([sample_category] + translation.tolist() + rotation.tolist())

        else:
            for j in range(translation_samples):
                translation = [
                    random.uniform(-1.0, 1.0),
                    random.uniform(-1.0, 1.0),
                    random.uniform(-1.0, 1.0)]
                rotation = [
                    random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0),
                    random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
                data.append([sample_category] + translation + rotation)

    columns = ['Category',
               'Translation X', 'Translation Y', 'Translation Z',
               'Rotation X', 'Rotation Y', 'Rotation Z', 'Rotation W'
               ]
    return pd.DataFrame(data, columns=columns)


def plot_histogram_by_category(data_frame):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data_frame, x='Distance', hue='Category', multiple='stack')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.title('Histogram of Distances by Category')
    plt.legend(title='Category')
    plt.show()


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

def calculate_distance_to_mean(frame):
    translation_columns = ['Translation X', 'Translation Y', 'Translation Z']
    mean_translation = frame[translation_columns].mean()
    distances = np.linalg.norm(frame[translation_columns] - mean_translation, axis=1)
    result_frame = pd.DataFrame({'Category': frame['Category'], 'Distance': distances})
    return result_frame


def plot_std_deviation(std_deviation_frame, target='Translation X'):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=std_deviation_frame.reset_index(), x='Category', y=target)
    plt.xlabel('Category')
    plt.ylabel('Standard Deviation')
    plt.title('Standard Deviation of Columns by Category')
    plt.show()


def calculate_standard_deviation_by_category(data_frame):
    std_deviation_frame = data_frame.groupby('Category').std()
    return std_deviation_frame


# Example usage:
num_categories_samples = 5
num_translations_samples = 10

df = create_sample_data(num_categories_samples, num_translations_samples)
# frame_std = calculate_standard_deviation_by_category(df)
# print(frame_std)
# plot_std_deviation(frame_std)
# print(df)
frame_distance = calculate_distance_to_mean(df)
print(frame_distance)
# plot_histogram_by_category(frame_distance)
plot_prop(frame_distance)
# plot_3d_scatter(df)

