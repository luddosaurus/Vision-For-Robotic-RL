#!/usr/bin/python3.8
import rospy
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg


class FixedTFBroadcaster:

    def __init__(self):
        self.pub_tf_left = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=10)
        self.pub_tf_right = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=10)

        while not rospy.is_shutdown():
            # Run this loop at about 10Hz
            rospy.sleep(0.1)

            t_left = geometry_msgs.msg.TransformStamped()
            t_left.header.frame_id = "panda_hand"
            t_left.header.stamp = rospy.Time.now()
            t_left.child_frame_id = "left_aruco"
            t_left.transform.translation.x = 0.036
            t_left.transform.translation.y = 0.055
            t_left.transform.translation.z = 0.04

            t_left.transform.rotation.x = -0.5
            t_left.transform.rotation.y = 0.5
            t_left.transform.rotation.z = -0.5
            t_left.transform.rotation.w = 0.5

            t_right = geometry_msgs.msg.TransformStamped()
            t_right.header.frame_id = "panda_hand"
            t_right.header.stamp = rospy.Time.now()
            t_right.child_frame_id = "right_aruco"
            t_right.transform.translation.x = 0.036
            t_right.transform.translation.y = -0.055
            t_right.transform.translation.z = 0.04

            t_right.transform.rotation.x = -0.5
            t_right.transform.rotation.y = 0.5
            t_right.transform.rotation.z = -0.5
            t_right.transform.rotation.w = 0.5

            tfm_left = tf2_msgs.msg.TFMessage([t_left])
            tfm_right = tf2_msgs.msg.TFMessage([t_right])
            self.pub_tf_left.publish(tfm_left)
            self.pub_tf_right.publish(tfm_right)


if __name__ == '__main__':
    rospy.init_node('fixed_tf2_broadcaster_left')
    tfb = FixedTFBroadcaster()

    rospy.spin()
