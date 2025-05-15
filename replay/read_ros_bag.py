import time
import os
import sys
from glob import glob
import geometry_msgs
import rosbag
import gc
import numpy as np
# from surgical_robotics_challenge.utils.task3_init import NeedleInitialization
dynamic_path = os.path.abspath(__file__ + "/../../")
# data_path = os.path.abspath(__file__+"/../../../../")
# print(dynamic_path)
sys.path.append(dynamic_path)
from scipy.spatial.transform import Rotation as R
import pickle


# def gripper_msg_to_jaw(msg):
#     '''
#     Map the MTM input to the gripper
#     '''
#     min = -0.698  # ~ -40 deg
#     max = 1.047  # ~60 deg
#     input_val = get_input_in_range(msg.position[0], min, max)
#     jaw_angle = (input_val - min) / (max - min)
#     return jaw_angle


if __name__ == '__main__':
    rosbag_folder = os.path.join(dynamic_path, 'record_bags')
    pre_fix = 'test'
    exp_name = 'init'
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

    # simulation_manager = SimulationManager('record_test')
    # time.sleep(0.5)
    # w = simulation_manager.get_world_handle()
    # time.sleep(0.2)
    # w.reset_bodies()
    # time.sleep(0.2)
    # cam = ECM(simulation_manager, 'CameraFrame')
    # # cam.servo_jp([0.0, 0.05, -0.01, 0.0])
    # time.sleep(0.5)
    # psm1 = PSM(simulation_manager, 'psm1', add_joint_errors=False)
    # time.sleep(0.5)
    # psm2 = PSM(simulation_manager, 'psm2', add_joint_errors=False)
    # time.sleep(0.5)
    # needle = simulation_manager.get_obj_handle('Needle')
    # time.sleep(0.2)
    #
    # needle_pose_list = []
    #
    # total_num = min(len(psm1_pos_ambf), len(psm2_pos_ambf), len(psm1_jaw_ambf), len(psm2_jaw_ambf))
    # # total_num = min(len(psm1_pos), len(psm2_pos), len(psm1_jaw), len(psm2_jaw))
    # print(total_num)
    # for i in range(total_num):
    #     # cam.servo_jp(ecm_pos[i])
    #     psm1.servo_jp(psm1_pos_ambf[i])
    #     psm1.set_jaw_angle(psm1_jaw_ambf[i])
    #     psm2.servo_jp(psm2_pos_ambf[i])
    #     psm2.set_jaw_angle(psm2_jaw_ambf[i])
    #
    #     # psm1.servo_jp(psm1_pos[i])
    #     # psm1.set_jaw_angle(psm1_jaw[i])
    #     # psm2.servo_jp(psm2_pos[i])
    #     # psm2.set_jaw_angle(psm2_jaw[i])
    #
    #     # needle_pose_item = needle.get_pose()
    #     # needle_pose_list.append(needle_pose_item)
    #
    #     time.sleep(0.01)
    #     count += 1
    #     sys.stdout.write(f'\r Running progress {count}/{total_num}')
    #     sys.stdout.flush()

    print('Done')