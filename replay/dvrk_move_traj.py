import os
import sys
import numpy as np
import crtk
import PyKDL
import rospy
import sys
import tf_conversions.posemath as pm
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA
import pickle
from std_msgs.msg import Bool
import math
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
import cv2
import rosbag
import gc
import json

dynamic_path = os.path.abspath(__file__ + "/../../")
# print(dynamic_path)
sys.path.append(dynamic_path)


def setting_arms_state(arm):
    if arm.operating_state() == "DISABLED":
        arm.enable()
        arm.home()


def checking_arm_state(arm):
    arm.check_connections()
    # make sure the arm is powered
    # print('-- Enabling arm')
    if not arm.enable(10):
        sys.exit('-- Failed to enable within 10 seconds')

    # print('-- Homing arm')
    if not arm.home(10):
        sys.exit('-- Failed to home within 10 seconds')


def FrameRotation2list(Frame_R):
    R_list = []
    for i in range(3):
        for j in range(3):
            R_list.append(Frame_R[i, j])
    return R_list


def FrameTranslation2list(Frame_t):
    t_list = []
    for i in range(3):
        t_list.append(Frame_t[i])
    return t_list


def rotation_interpolation(init_rotation, target_rotation):
    from scipy.spatial.transform import Rotation as Rot
    r = Rot.from_matrix

#### need to revise if get rid of compressed image
class VideoGrabber:
    def __init__(self, frame_topic):
        self.bridge = CvBridge()
        self.image = None
        # rospy.Subscriber(name=frame_topic, data_class=CompressedImage, callback=self.callback, queue_size=1,
        #                  buff_size=2 ** 18)
        rospy.Subscriber(frame_topic, Image, self.callback, queue_size=1)

    def callback(self, data):
        # self.image = self.bridge.compressed_imgmsg_to_cv2(data, 'passthrough')
        self.image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

    def extract_image(self, image_path):
        cv2.imwrite(image_path, self.image)
        # print('frame saved!')


class arm_psm:
    # simplified jaw class to close gripper
    class __Jaw:
        def __init__(self, ral, expected_interval, operating_state_instance):
            self.__crtk_utils = crtk.utils(self, ral, expected_interval, operating_state_instance)
            self.__crtk_utils.add_move_jp()
            self.__crtk_utils.add_servo_jp()
            self.__crtk_utils.add_servo_jf()
            self.__crtk_utils.add_setpoint_js()
    class __local:
        def __init__(self, local_ral, expected_interval):
            self.__crtk_utils = crtk.utils(self, local_ral, expected_interval)
            self.__crtk_utils.add_measured_cp()
            self.__crtk_utils.add_measured_cv()
        
    def __init__(self, ral, device_namespace, expected_interval):
        # ROS initialization
        if not rospy.get_node_uri():
            # rospy.init_node('simplified_arm_class', anonymous = False, log_level = rospy.WARN)
            rospy.init_node('si_arm_class', anonymous=False, log_level=rospy.WARN)
        # populate this class with all the ROS topics we need
        self.__ral = ral.create_child(device_namespace)
        self.__crtk_utils = crtk.utils(self, self.__ral, expected_interval)
        self.__crtk_utils.add_operating_state()
        self.__crtk_utils.add_servo_jp()
        self.__crtk_utils.add_servo_jf()
        self.__crtk_utils.add_move_jp()
        self.__crtk_utils.add_move_cp()
        self.__crtk_utils.add_measured_js()
        self.__crtk_utils.add_setpoint_js()
        self.__crtk_utils.add_setpoint_cp()
        self.__crtk_utils.add_measured_cp()
        self.__crtk_utils.add_measured_cv()
        jaw_ral = self.ral().create_child('/jaw')
        self.jaw = self.__Jaw(jaw_ral, expected_interval,
                              operating_state_instance=self)
        local_ral = self.ral().create_child('/local')
        self.local = self.__local(local_ral, expected_interval)


        # self.local = self.__ral.create_child('local')
        # self.__crtk_utils_local = crtk.utils(self, self.local, expected_interval)
        # self.__crtk_utils_local.add_measured_cp()



    def ral(self):
        return self.__ral

    def check_connections(self, timeout=5.0):
        self.__ral.check_connections(timeout)


class arm_ecm:
    class __local:
        def __init__(self, local_ral, expected_interval):
            self.__crtk_utils = crtk.utils(self, local_ral, expected_interval)
            self.__crtk_utils.add_measured_cp()
            self.__crtk_utils.add_measured_cv()

    def __init__(self, ral, device_namespace, expected_interval):
        # ROS initialization
        if not rospy.get_node_uri():
            # rospy.init_node('simplified_arm_class', anonymous = False, log_level = rospy.WARN)
            rospy.init_node('si_arm_class', anonymous=False, log_level=rospy.WARN)
        # populate this class with all the ROS topics we need
        self.__ral = ral.create_child(device_namespace)
        self.__crtk_utils = crtk.utils(self, self.__ral, expected_interval)
        self.__crtk_utils.add_operating_state()
        self.__crtk_utils.add_servo_jp()
        self.__crtk_utils.add_servo_jf()
        self.__crtk_utils.add_move_jp()
        self.__crtk_utils.add_measured_js()
        self.__crtk_utils.add_setpoint_js()
        self.__crtk_utils.add_setpoint_cp()
        self.__crtk_utils.add_measured_cp()
        self.__crtk_utils.add_measured_cv()

        local_ral = self.ral().create_child('/local')
        self.local = self.__local(local_ral, expected_interval)

    def ral(self):
        return self.__ral

    def check_connections(self, timeout=5.0):
        self.__ral.check_connections(timeout)


if __name__ == '__main__':
    rosbag_folder = os.path.join(dynamic_path, 'record_bags', 'ros1')
    pre_fix = 'test'
    exp_name = '0'
    rosbag_name = os.path.join(rosbag_folder, f'{pre_fix}_{exp_name}.bag')

    print(rosbag_name)

    bag = rosbag.Bag(rosbag_name)
    topics = list(bag.get_type_and_topic_info()[1].keys())
    types = [val[0] for val in bag.get_type_and_topic_info()[1].values()]

    count = 0
    ecm_pos = []
    psm1_pos = []
    psm2_pos = []
    t_psm1 = []
    t_psm2 = []
    t_ecm = []
    psm1_jaw = []
    psm2_jaw = []
    needle_pos = []

    ## ambf raw replay
    for topic, msg, t in bag.read_messages(topics='/PSM1/setpoint_js'):
        psm1_pos_temp = [msg.position[0],
                         msg.position[1],
                         msg.position[2],
                         msg.position[3],
                         msg.position[4],
                         msg.position[5]]
        psm1_pos.append(psm1_pos_temp)
        t_psm1.append(t)
        count += 1
    print('psm1 ambf record count: ', count)
    count = 0

    for topic, msg, t in bag.read_messages(topics='/PSM2/setpoint_js'):
        psm2_pos_temp = [msg.position[0],
                         msg.position[1],
                         msg.position[2],
                         msg.position[3],
                         msg.position[4],
                         msg.position[5]]
        psm2_pos.append(psm2_pos_temp)
        t_psm2.append(t)
        count += 1
    print('psm2 ambf record count: ', count)
    count = 0

    for topic, msg, t in bag.read_messages(topics='/ECM/setpoint_js'):
        ecm_pos_temp = [msg.position[0],
                        msg.position[1],
                        msg.position[2],
                        msg.position[3]]
        ecm_pos.append(ecm_pos_temp)
        t_ecm.append(t)
        count += 1
    print('ecm ambf record count: ', count)
    count = 0

    for topic, msg, t in bag.read_messages(topics='/PSM1/jaw/setpoint_js'):
        psm1_jaw_temp = [msg.position[0]]
        psm1_jaw.append(psm1_jaw_temp)
        count += 1
    print('PSM1 gripper record count: ', count)
    count = 0

    for topic, msg, t in bag.read_messages(topics='/PSM2/jaw/setpoint_js'):
        psm2_jaw_temp = [msg.position[0]]
        psm2_jaw.append(psm2_jaw_temp)
        count += 1
    print('PSM2 gripper record count: ', count)
    count = 0
    gc.collect()

    total_num = min(len(psm1_pos), len(psm2_pos), len(ecm_pos), len(psm1_jaw), len(psm2_jaw))
    print('Total num: ', total_num)

    ### Can modify the image topics
    # left_frame_topic = '/test/left/image_raw/compressed'
    # right_frame_topic = '/test/right/image_raw/compressed'
    # side_frame_topic = '/sidecam/image_raw/compressed'
    left_frame_topic = '/test/left/image_raw'
    right_frame_topic = '/test/right/image_raw'
    side_frame_topic = '/sidecam/image_raw'

    print("Initializing arms...")
    ral = crtk.ral('dvrk_si_test')

    psm1 = arm_psm(ral, 'PSM1', 0.1)
    psm2 = arm_psm(ral, 'PSM2', 0.1)
    ecm = arm_ecm(ral, 'ECM', 0.1)

    setting_arms_state(psm1)
    setting_arms_state(psm2)
    setting_arms_state(ecm)

    # Initialize video grabber
    left_video_grabber = VideoGrabber(left_frame_topic)
    right_video_grabber = VideoGrabber(right_frame_topic)
    side_video_grabber = VideoGrabber(side_frame_topic)

    input('---> Move the arm to initial position, Press \"Enter\" to start')

    # Move to the initial pose

    js_init_psm1 = np.array(psm1_pos[0])
    js_init_psm2 = np.array(psm2_pos[0])
    js_init_ecm = np.array(ecm_pos[0])

    ecm.move_jp(js_init_ecm).wait()
    psm1.move_jp(js_init_psm1).wait()
    psm2.move_jp(js_init_psm2).wait()

    input('---> Move the jaw to initial positions, Press \"Enter\" to start')

    jaw_init_psm1 = np.array(psm1_jaw[0])
    jaw_init_psm2 = np.array(psm2_jaw[0])

    psm1.jaw.move_jp(jaw_init_psm1).wait()
    psm2.jaw.move_jp(jaw_init_psm2).wait()

    input('---> Ready to start, Press \"Enter\" to start')

    # create frame folders
    left_image_path = './left_frames'
    right_image_path = './right_frames'
    side_image_path = './side_frames'
    api_cp_path = './api_cp_files'
    api_jp_path = './api_jp_files'
    os.makedirs(left_image_path, exist_ok=True)
    os.makedirs(right_image_path, exist_ok=True)
    os.makedirs(side_image_path, exist_ok=True)
    os.makedirs(api_cp_path, exist_ok=True)
    os.makedirs(api_jp_path, exist_ok=True)

    # Move the robot according to the trajectory
    for idx in range(total_num):
        psm1.jaw.move_jp(np.array(psm1_jaw[idx])).wait()
        psm2.jaw.move_jp(np.array(psm2_jaw[idx])).wait()

        psm1.move_jp(np.array(psm1_pos[idx])).wait()
        psm2.move_jp(np.array(psm2_pos[idx])).wait()


        # 2. save images
        left_video_grabber.extract_image(os.path.join(left_image_path, f'frame{idx}.png'))
        right_video_grabber.extract_image(os.path.join(right_image_path, f'frame{idx}.png'))
        side_video_grabber.extract_image(os.path.join(side_image_path, f'frame{idx}.png'))

        # 3. record api data
        cp_temp_dict = {}
        jp_temp_dict = {}

        psm1_cp, _ = psm1.measured_cp()
        psm1_R = FrameRotation2list(psm1_cp.M)
        psm1_t = FrameTranslation2list(psm1_cp.p)
        cp_temp_dict["PSM1"] = {"R": psm1_R, "t": psm1_t}

        psm1_local_cp, _ = psm1.local.measured_cp()
        psm1_local_R = FrameRotation2list(psm1_local_cp.M)
        psm1_local_t = FrameTranslation2list(psm1_local_cp.p)
        cp_temp_dict["PSM1_local"] = {"R": psm1_local_R, "t": psm1_local_t}

        psm1_cv, _ = psm1.measured_cv()
        psm1_linear = list(psm1_cv[0:3])
        psm1_angular = list(psm1_cv[3:6])
        cp_temp_dict["PSM1_cv"] = {"linear": psm1_linear, "angular": psm1_angular}

        psm2_cp, _ = psm2.measured_cp()
        psm2_R = FrameRotation2list(psm2_cp.M)
        psm2_t = FrameTranslation2list(psm2_cp.p)
        cp_temp_dict["PSM2"] = {"R": psm2_R, "t": psm2_t}

        psm2_local_cp, _ = psm2.local.measured_cp()
        psm2_local_R = FrameRotation2list(psm2_local_cp.M)
        psm2_local_t = FrameTranslation2list(psm2_local_cp.p)
        cp_temp_dict["PSM2_local"] = {"R": psm2_local_R, "t": psm2_local_t}

        psm2_cv, _ = psm2.measured_cv()
        psm2_linear = list(psm2_cv[0:3])
        psm2_angular = list(psm2_cv[3:6])
        cp_temp_dict["PSM2_cv"] = {"linear": psm2_linear, "angular": psm2_angular}

        ecm_cp, _ = ecm.measured_cp()
        ecm_R = FrameRotation2list(ecm_cp.M)
        ecm_t = FrameTranslation2list(ecm_cp.p)
        cp_temp_dict["ECM"] = {"R": ecm_R, "t": ecm_t}

        ecm_local_cp, _ = ecm.local.measured_cp()
        ecm_local_R = FrameRotation2list(ecm_local_cp.M)
        ecm_local_t = FrameTranslation2list(ecm_local_cp.p)
        cp_temp_dict["ECM_local"] = {"R": ecm_local_R, "t": ecm_local_t}

        psm1_jp, _ = psm1.measured_jp()
        psm2_jp, _ = psm2.measured_jp()
        ecm_jp, _ = ecm.measured_jp()
        jp_temp_dict = {"PSM1": psm1_jp.tolist(), "PSM2": psm2_jp.tolist(), "ECM": ecm_jp.tolist()}

        # psm1_cv, _ = psm1.measured_cv() ## 6x1 ndarray, first 3 linear, the other 3 angular
        #
        # psm1_scp, _ = psm1.setpoint_cp() ## same as measured_cp()
        #
        # psm1_sjs = psm1.setpoint_js() ### tuple 4 (pos, vel, effort, time); pos/ve/eff 6x1 ndarray
        # print(psm1_sjs)
        # print(type(psm1_sjs))
        sys.stdout.write('\r-- Progress: frame %s / %s' % (idx+1, total_num))
        sys.stdout.flush()
        f_cp = os.path.join(api_cp_path, f'frame{idx}.json')
        f_jp = os.path.join(api_jp_path, f'frame{idx}.json')

        with open(f_cp, 'w') as fcp:
            json.dump(cp_temp_dict, fcp)

        with open(f_jp, 'w') as fjp:
            json.dump(jp_temp_dict, fjp)

        gc.collect()