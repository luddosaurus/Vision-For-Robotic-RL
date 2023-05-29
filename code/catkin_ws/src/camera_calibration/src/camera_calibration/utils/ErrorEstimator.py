import numpy as np
import pandas as pd


class ErrorEstimator:

    @staticmethod
    def calculate_distance_to_truth(frame, truth_translation):
        translation_columns = ['Translation X', 'Translation Y', 'Translation Z']
        translations = frame[translation_columns]

        distances = np.linalg.norm(translations - truth_translation, axis=1)

        result_frame = pd.DataFrame({'Category': frame['Category'], 'Distance': distances})
        return result_frame

    @staticmethod
    def calculate_standard_deviation_by_category(data_frame):
        std_deviation_frame = data_frame.groupby('Category').std()
        return std_deviation_frame

    @staticmethod
    def calculate_variance_by_category(dataframe):
        # Group the dataframe by "Category" and calculate variance for each group
        grouped = dataframe.groupby("Category")
        variance_frame = grouped.var(ddof=0).fillna(0)

        return variance_frame


































