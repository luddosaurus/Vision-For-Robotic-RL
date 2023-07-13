# ROS
import rospy
from sensor_msgs.msg import Image
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg
import tf.transformations as tf


class TFPublish:

    @staticmethod
    def publish_static_stamped_transform(publisher, transform_stamped, parent_name, child_name):
        # Message
        transform_stamped_msg = geometry_msgs.msg.TransformStamped()

        # Info
        transform_stamped_msg.header.stamp = rospy.Time.now()
        transform_stamped_msg.header.frame_id = parent_name
        transform_stamped_msg.child_frame_id = child_name

        # Test this!
        # transform_stamped_msg.header.stamp = rospy.Time.now()
        # transform_stamped_msg.header.frame_id = transform_stamped.header.frame_id
        # transform_stamped_msg.child_frame_id = transform_stamped.child_frame_id

        transform = transform_stamped.transform
        # Data
        transform_stamped_msg.transform.translation.x = transform.translation.x
        transform_stamped_msg.transform.translation.y = transform.translation.y
        transform_stamped_msg.transform.translation.z = transform.translation.z

        transform_stamped_msg.transform.rotation.x = transform.rotation.x
        transform_stamped_msg.transform.rotation.y = transform.rotation.y
        transform_stamped_msg.transform.rotation.z = transform.rotation.z
        transform_stamped_msg.transform.rotation.w = transform.rotation.w

        # tfm = tf2_msgs.msg.TFMessage([transform_stamped_msg])
        publisher.sendTransform(transform_stamped_msg)
        # publisher.publish(tfm)

    @staticmethod
    def publish_transform(publisher, parent_name, child_name, translation, rotation):
        # Message
        transform_stamped_msg = geometry_msgs.msg.TransformStamped()

        # Info
        transform_stamped_msg.header.stamp = rospy.Time.now()
        transform_stamped_msg.header.frame_id = parent_name
        transform_stamped_msg.child_frame_id = child_name

        # Data
        transform_stamped_msg.transform.translation.x = translation[0]
        transform_stamped_msg.transform.translation.y = translation[1]
        transform_stamped_msg.transform.translation.z = translation[2]
        transform_stamped_msg.transform.rotation.x = rotation[0]
        transform_stamped_msg.transform.rotation.y = rotation[1]
        transform_stamped_msg.transform.rotation.z = rotation[2]
        transform_stamped_msg.transform.rotation.w = rotation[3]

        tfm = tf2_msgs.msg.TFMessage([transform_stamped_msg])

        publisher.sendTransform(transform_stamped_msg)

    @staticmethod
    def publish_static_transform(publisher, parent_name, child_name, translation, rotation):
        # Message
        transform_stamped_msg = geometry_msgs.msg.TransformStamped()

        # Info
        transform_stamped_msg.header.stamp = rospy.Time.now()
        transform_stamped_msg.header.frame_id = parent_name
        transform_stamped_msg.child_frame_id = child_name

        # Data
        transform_stamped_msg.transform.translation.x = translation[0]
        transform_stamped_msg.transform.translation.y = translation[1]
        transform_stamped_msg.transform.translation.z = translation[2]
        transform_stamped_msg.transform.rotation.x = rotation[0]
        transform_stamped_msg.transform.rotation.y = rotation[1]
        transform_stamped_msg.transform.rotation.z = rotation[2]
        transform_stamped_msg.transform.rotation.w = rotation[3]

        # tfm = tf2_msgs.msg.TFMessage([transform_stamped_msg])
        publisher.sendTransform(transform_stamped_msg)
