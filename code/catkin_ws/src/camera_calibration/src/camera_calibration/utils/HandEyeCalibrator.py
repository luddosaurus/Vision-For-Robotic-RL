import numpy as np
import cv2


class HandEyeCalibrator:

    # find offset between end-effector and aruco markers
    @staticmethod
    def hand_eye_calibration(robot_poses, camera_poses):
        """
        Parameters:
        robot_poses (list): A list of 4x4 homogeneous transformation matrices
            representing the end-effector poses in the robot base frame.
        camera_poses (list): A list of 4x4 homogeneous transformation matrices
            representing the ArUco marker poses in the camera frame.

        Returns:
        T_camera_endeffector (numpy array): A 4x4 homogeneous transformation
            matrix representing the offset between the camera and the end-effector.
        """
        # Convert the pose lists to numpy arrays
        robot_poses = np.array(robot_poses)
        camera_poses = np.array(camera_poses)

        # Compute the A and B matrices
        A = np.zeros((3 * len(robot_poses), 3))
        B = np.zeros((3 * len(robot_poses), 1))
        for i in range(len(robot_poses)):
            Ai = np.eye(3) - robot_poses[i][:3, :3]
            Bi = robot_poses[i][:3, 3] - np.dot(Ai, camera_poses[i][:3, 3])

            Bi = np.reshape(Bi, (3,1))

            A[3 * i:3 * i + 3, :] = Ai
            B[3 * i:3 * i + 3, :] = Bi

        # Solve for the quaternion and translation
        ATA = np.dot(A.T, A)
        ATB = np.dot(A.T, B)
        q = np.linalg.solve(ATA, ATB)
        q /= np.linalg.norm(q)
        t = np.zeros((3, 1))
        print(q)
        for i in range(len(robot_poses)):
            print(robot_poses[0][3, :3].shape)
            t += np.dot(robot_poses[i][3, :3],
                        camera_poses[i][3, :3] - np.dot(cv2.Rodrigues(q)[0],
                        camera_poses[i][3, :3].T.dot(robot_poses[i][3, :3])))
        t /= len(robot_poses)

        # Convert the quaternion and translation to a homogeneous transformation matrix
        T_camera_endeffector = np.eye(4)
        T_camera_endeffector[:3, :3] = cv2.Rodrigues(q)[0]
        T_camera_endeffector[:3, 3] = t.T

        return T_camera_endeffector



