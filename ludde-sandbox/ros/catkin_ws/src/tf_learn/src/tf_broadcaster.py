#! /home/dat14lja/Desktop/Thesis/Vision-For-Robotic-RL/ludde-sandbox/venv/bin/python

import rospy
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg


class FixedTFBroadcaster:

    def __int__(self):
        self.pub_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)

        while not rospy.is_shutdown():
            rospy.sleep(0.1)

            t = geometry_msgs.msg.TransformStamped()
            t.header.frame_id = "turtle"
            t.header.stamp = rospy.Time.now()
            t.child_frame_id = "carrot"
            t.transform.translation.x = 0.0
            t.transform.translation.y = 2.0
            t.transform.translation.z = 0.0

            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            tf_message = tf2_msgs.msg.TFMessage([t])
            self.pub_tf.publish(tf_message)


if __name__ == '__main__':
    rospy.init_node('tf_broadcast_node')
    tfb = FixedTFBroadcaster()

    rospy.spin()
