#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from pdb import set_trace as pause
import argparse
import struct
import sys
import rospy
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
from trac_ik_baxter.srv import GetConstrainedPositionIK

class baxter_qdes_publisher():
    def __init__(self):
        rospy.init_node('bax_q_des', anonymous=True)
        self.trac_ik_serv = rospy.ServiceProxy('trac_ik_right', GetConstrainedPositionIK, persistent=True)
        self.des_pose = None
        rospy.Subscriber('/bax_des_right_pose', PoseStamped, self.desired_pose_callback) #subcribe to desired pose for baxter's right arm
        self.qdes_publisher =  rospy.Publisher('/qdes', Float32MultiArray, queue_size = 1)
        self.qdes = Float32MultiArray()
        self.rate = rospy.Rate(50)

    def desired_pose_callback(self, msg):
        self.des_pose = msg

    # this is the default baxter service call which fails a lot
    def run_ik(self):
        limb = 'right'
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        ikreq = SolvePositionIKRequest()
        while self.des_pose == None:
            print 'No desired pose received'
        ikreq.pose_stamp.append(self.des_pose)
        try:
            rospy.wait_for_service(ns, 2.0)
            resp = iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return 1

        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        if (resp_seeds[0] == resp.RESULT_INVALID):
            print("INVALID POSE - No Valid Joint Solution Found.")

        # print resp.joints[0].position
        return resp.joints[0].position # a tuple of the joint angles

    # this runs trac_ik for inverse kinematics which is more robust
    def run_trac_ik(self):
        while self.des_pose == None:
            print 'No desired pose received'
        result = self.trac_ik_serv([self.des_pose],[], 1.e-2, 50)
        return result.joints[0].position

    def publisher(self, jangles):
        # jangles is a list of joint angles for baxters right arm
        # jangles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.qdes.data = np.array(jangles)
        self.qdes_publisher.publish(self.qdes)
        print self.qdes

if __name__=='__main__':
    # start_time = rospy.get_time()
    # joint_angles = run_ik('right', des_pose_bax_right)
    # jangles = list(joint_angles)
    baxter = baxter_qdes_publisher()
    while not rospy.is_shutdown():
        # print des_pose
        # joint_angles = baxter.run_ik()
        jangles = list(baxter.run_trac_ik())
        print jangles
        # jangles = list(joint_angles)
        # baxter.publisher(jangles)
        baxter.rate.sleep()

