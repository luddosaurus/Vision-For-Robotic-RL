import numpy as np
import pandas as pd

from camera_calibration.utils.EyeHandSolver import EyeHandSolver
from camera_calibration.utils.HarryPlotterAndTheChamberOfSeaborn import HarryPlotter
from camera_calibration.utils.TypeConverter import TypeConverter
from camera_calibration.utils.ErrorEstimator import ErrorEstimator


class ExtrinsicEvaluator:
    TYPE_ORDER = 0
    TYPE_RANDOM_AVG = 1

    def __init__(self, intrinsic_camera, distortion, camera2charuco, hand2world):
        self.intrinsic_camera = intrinsic_camera
        self.distortion = distortion
        self.camera2charuco = camera2charuco
        self.hand2world = hand2world
        self.extrinsic_solver = EyeHandSolver(transforms_hand2world=self.hand2world,
                                              transforms_camera2charuco=self.camera2charuco
                                              )

    def __solve_all_combos(self):
        # Solve all versions
        pose_estimation_samples_order = self.extrinsic_solver.solve_sample_sizes()
        pose_estimations_samples_random = self.extrinsic_solver.solve_all_sample_combos()
        pose_estimations_methods = self.extrinsic_solver.solve_all_algorithms()
        pose_estimations_method_samples = self.extrinsic_solver.solve_all_method_samples()

    def evaluate2d(self, evaluation_type=TYPE_ORDER, title="2D Plot"):
        if evaluation_type == self.TYPE_ORDER:
            self.__evaluate_order(title)
        if evaluation_type == self.TYPE_RANDOM_AVG:
            self.__evaluate_random_average(title)
        else:
            print("Unknown evaluation type")

    def __evaluate_order(self, title):

        # Solve
        pose_dict = self.extrinsic_solver.solve_sample_sizes()

        # Panda Frame [Category, Translation XYZ, Rotation XYZW]
        frame_samples = TypeConverter.convert_to_dataframe(pose_dict, category='Sample Sizes')

        # Plot
        ExtrinsicEvaluator.__plot2d(frame_samples, x='Sample Sizes', title=title)

    def __evaluate_random_average(self, title):

        # Solve
        pose_dict = self.extrinsic_solver.solve_sample_sizes()

        # Panda Frame [Category, Translation XYZ, Rotation XYZW]
        frame_samples = TypeConverter.convert_to_dataframe(pose_dict, category='Samples')
        frame_samples.groupby('Samples').mean()
        frame_samples.rename(columns={'Samples': 'Average for Sample Size'}, inplace=True)

        # Plot
        ExtrinsicEvaluator.__plot2d(frame_samples, x='Average for Sample Size', title=title)

    @staticmethod
    def __plot2d(frame, x, title="", ):
        HarryPlotter.plot_line(frame, y='Translation X', x=x)
        HarryPlotter.plot_line(frame, y='Translation Y', x=x)
        HarryPlotter.plot_line(frame, y='Translation Z', x=x)

    @staticmethod
    def __old_plot(frame):
        # Standard Deviations
        frame_std = ErrorEstimator.calculate_standard_deviation_by_category(frame)

        # Each value
        HarryPlotter.plot_prop(frame, x='Translation X')
        HarryPlotter.plot_prop(frame, x='Translation Y')
        HarryPlotter.plot_prop(frame, x='Translation Z')

        # Distance density (World Center to Pose estimations)
        translation_columns = ["Translation X", "Translation Y", "Translation Z"]
        HarryPlotter.plot_distance_density(frame, translation_columns)

        rotation_columns = ["Rotation X", "Rotation Y", "Rotation Z", "Rotation W"]
        HarryPlotter.plot_distance_density(frame, rotation_columns)

        # Variance
        frame_variance = ErrorEstimator.calculate_variance_by_category(frame)
        HarryPlotter.stacked_histogram(frame_variance)

    @staticmethod
    def __plot3d(frame):
        HarryPlotter.plot_3d_scatter(frame)
