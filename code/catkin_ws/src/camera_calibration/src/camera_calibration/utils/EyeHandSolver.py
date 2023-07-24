from itertools import combinations
import random

import cv2

from camera_calibration.utils.TypeConverter import TypeConverter


class EyeHandSolver(object):
    TSAI = 0
    PARK = 1
    HORAUD = 2
    ANDREFF = 3
    DANIILIDIS = 4

    def __init__(self, transforms_hand2world, transforms_camera2charuco):
        self.methods = [
            cv2.CALIB_HAND_EYE_TSAI,
            cv2.CALIB_HAND_EYE_PARK,
            cv2.CALIB_HAND_EYE_HORAUD,
            cv2.CALIB_HAND_EYE_ANDREFF,
            cv2.CALIB_HAND_EYE_DANIILIDIS
        ]
        self.num_images_to_capture = len(self.transforms_camera2charuco)
        self.transforms_hand2world = transforms_hand2world
        self.transforms_camera2charuco = transforms_camera2charuco

    @staticmethod
    def solve(fixed2attached, hand2base, solve_method, attached2hand_guess=None):
        # fixed = thing on table
        # attached = thing on arm
        # hand = gripper
        # bases = world
        # Solves AX=XB with hand2base being A and fixed2attached B

        # Fixed2Attached
        rot_fixed2attached, tran_fixed2attached = TypeConverter.transform_to_matrices(
            fixed2attached)

        # Hand2World
        rot_hand2world, tran_hand2world = TypeConverter.transform_to_matrices(
            hand2base)

        # Attached2Hand
        if attached2hand_guess is not None:
            # Init Guess Fixed2Hand
            rot_attached2hand_guess, trand_attached2hand_guess = TypeConverter.transform_to_matrices(
                attached2hand_guess
            )
            rot_attached2hand, tran_attached2hand = cv2.calibrateHandEye(
                R_gripper2base=rot_hand2world,
                t_gripper2base=tran_hand2world,
                R_target2cam=rot_fixed2attached,
                t_target2cam=tran_fixed2attached,
                R_cam2gripper=rot_attached2hand_guess,
                t_cam2gripper=trand_attached2hand_guess,
                method=solve_method
            )
        else:
            try:
                rot_attached2hand, tran_attached2hand = cv2.calibrateHandEye(
                    R_gripper2base=rot_hand2world,
                    t_gripper2base=tran_hand2world,
                    R_target2cam=rot_fixed2attached,
                    t_target2cam=tran_fixed2attached,
                    method=solve_method
                )
            except:
                print('bad value')
                return None, None

        return rot_attached2hand, tran_attached2hand

    def solve_sample_sizes(
            self,
            solve_method=None,
            start_sample_size=None,
            end_sample_size=None,
            step_size=1):
        """
        Solves for all sample sizes in order.
        @param start_sample_size:
        @param end_sample_size:
        @param solve_method: EyeHandSolver.TSAI ...
        @param step_size:
        @return: camera poses of type
        dictionary key[sample_size] value[list(tuple(rotation, translation))]
        """
        if end_sample_size is None:
            end_sample_size = self.num_images_to_capture + 1
        if solve_method is None:
            solve_method = self.methods[0]
        if start_sample_size is None:
            half_size = int(self.num_images_to_capture / 2)
            start_sample_size = half_size if half_size >= 3 else 3

        solve_method = self.methods[solve_method]
        poses = dict()
        list_size = len(self.transforms_camera2charuco)

        # For every sample size
        for sample_size in range(start_sample_size, end_sample_size, step_size):
            poses[sample_size] = list()

            camera2aruco_subset = self.transforms_camera2charuco[:sample_size]
            hand2base_subset = self.transforms_hand2world[:sample_size]

            # Do and save estimation
            rotation, translation = self.solve(
                fixed2attached=camera2aruco_subset,
                hand2base=hand2base_subset,
                solve_method=solve_method
            )
            if rotation is not None and translation is not None:
                poses[sample_size].append((rotation, translation))

        return poses

    def solve_all_sample_combos(
            self,
            solve_method=None,
            start_sample_size=None,
            end_sample_size=None,
            step_size=1,
            max_random_iterations=300):
        """
        @param solve_method:
        @param start_sample_size:
        @param end_sample_size:
        @param step_size:
        @param max_random_iterations:
        @return: camera poses of type
        dictionary key[sample_size] value[list(tuple(rotation, translation))]
        """
        if end_sample_size is None:
            end_sample_size = self.num_images_to_capture + 1
        if solve_method is None:
            solve_method = self.methods[0]
        if start_sample_size is None:
            half_size = int(self.num_images_to_capture / 2)
            start_sample_size = half_size if half_size >= 3 else 3

        poses = dict()
        list_size = len(self.transforms_camera2charuco)
        max_iterations = 0
        # For every sample size
        for sample_size in range(start_sample_size, end_sample_size, step_size):
            print(sample_size)
            poses[sample_size] = list()

            # For every index combination
            for sample_indices in combinations(range(list_size), sample_size):
                # Take out subset of indices
                camera2aruco_subset = [self.transforms_camera2charuco[index] for index in sample_indices]
                hand2base_subset = [self.transforms_hand2world[index] for index in sample_indices]

                # Do and save estimation
                rot, tran = self.solve(
                    fixed2attached=camera2aruco_subset,
                    hand2base=hand2base_subset,
                    solve_method=solve_method
                )
                if rot is not None and tran is not None:
                    poses[sample_size].append(
                        (rot, tran)
                    )
                max_iterations += 1
                if max_iterations >= max_random_iterations:
                    break
            max_iterations = 0

        return poses

    def solve_all_method_samples(
            self,
            start_sample_size=None,
            end_sample_size=None,
            step_size=1,
            max_random_iterations=300):

        # Solve all sample sizes for each algorithm
        if end_sample_size is None:
            end_sample_size = self.num_images_to_capture + 1
        if start_sample_size is None:
            half_size = int(self.num_images_to_capture / 2)
            start_sample_size = half_size if half_size >= 3 else 3
        poses = dict()
        max_iterations = 0
        for method in self.methods:
            poses[method] = list()

            for sample_size in range(start_sample_size, end_sample_size, step_size):
                sample_indices = random.sample(range(len(self.transforms_camera2charuco)), sample_size)
                camera2aruco_subset = [self.transforms_camera2charuco[index] for index in sample_indices]
                hand2base_subset = [self.transforms_hand2world[index] for index in sample_indices]

                poses[method].append(
                    self.solve(
                        fixed2attached=camera2aruco_subset,
                        hand2base=hand2base_subset,
                        solve_method=method
                    )
                )
                max_iterations += 1
                if max_iterations >= max_random_iterations:
                    break
            max_iterations = 0

        return poses

    def solve_all_algorithms(self):

        poses = dict()

        for method in self.methods:
            # if method == self.methods[3]:
            #     continue
            poses[method] = list()
            poses[method].append(
                self.solve(
                    fixed2attached=self.transforms_camera2charuco,
                    hand2base=self.transforms_hand2world,
                    solve_method=method
                )
            )

        return poses

    def update_transforms(self, transforms_hand2world, transforms_camera2charuco):
        self.transforms_hand2world = transforms_hand2world
        self.transforms_camera2charuco = transforms_camera2charuco
