import numpy as np
import cv2

class PoseQualityHelper:

    def measure_pose_consistency(transforms):
        rotations = []
        translations = []
        for transform in transforms:
            rot_quat = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            trans_vec = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ]
            rotations.append(rot_quat)
            translations.append(trans_vec)

        rotation_std = np.std(rotations, axis=0)
        translation_std = np.std(translations, axis=0)

        return rotation_std, translation_std
