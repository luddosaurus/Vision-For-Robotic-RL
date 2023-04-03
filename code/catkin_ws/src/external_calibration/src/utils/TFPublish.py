# ROS
import rospy
from sensor_msgs.msg import Image
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg
import tf.transformations as tf


def publish(publisher, parent_name, child_name, translation, rotation):

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
    publisher.publish(tfm)


def publish(publisher, transform, parent_name, child_name):
    # Message
    transform_stamped_msg = geometry_msgs.msg.TransformStamped()

    # Info
    transform_stamped_msg.header.stamp = rospy.Time.now()
    transform_stamped_msg.header.frame_id = parent_name
    transform_stamped_msg.child_frame_id = child_name

    # Data
    transform_stamped_msg.transform.translation = transform.translation
    transform_stamped_msg.transform.rotation = transform.rotation

    tfm = tf2_msgs.msg.TFMessage([transform_stamped_msg])
    publisher.publish(tfm)
