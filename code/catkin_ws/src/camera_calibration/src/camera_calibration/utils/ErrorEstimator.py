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


































