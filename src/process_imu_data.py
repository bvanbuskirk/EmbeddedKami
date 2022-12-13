#!/usr/bin/python3
import matplotlib.pyplot as plt
from matplotlib import animation as anim, cm
import rospy
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import collections
from pdb import set_trace
import wandb

from std_msgs.msg import Header
from ros_stuff.msg import ImuData, ProcessedData

RAD_TO_DEG = 180 / 3.14

class Laptop():
    def __init__(self, robot_ids, plot=False):
        rospy.init_node("laptop")
        rospy.Subscriber("/imu_data", ImuData, self.receive_imu_data, queue_size=10)
        print("IMU subscriber initialized")

        self.publisher = rospy.Publisher("/transformed_imu_data", ProcessedData, queue_size=1)

        self.plot = plot
        self.robots = robot_ids
        self.num_robots = len(self.robots) #if isinstance(self.robots, list) else 1
        self.ac_num = [-1 for i in range(self.num_robots)]
        self.first_timestamp = [None for i in range(self.num_robots)]
        self.latest_timestamp = [None for i in range(self.num_robots)]
        self.began = [False for i in range(self.num_robots)]
        # map for robot_id to data index
        self.robot_idx_map = {self.robots[i]: i for i in range(self.num_robots)}

        # databases and buffers for each robot
        self.data = [{} for i in range(self.num_robots)]
        self.proc_data = [{} for i in range(self.num_robots)]
        # 6 axes of data
        self.ax_buffer = [[] for i in range(self.num_robots)]
        self.ay_buffer = [[] for i in range(self.num_robots)]
        self.gx_buffer = [[] for i in range(self.num_robots)]
        self.gy_buffer = [[] for i in range(self.num_robots)]
        self.mx_buffer = [[] for i in range(self.num_robots)]
        self.my_buffer = [[] for i in range(self.num_robots)]
        # self.a_thresh = 0.5

        if self.plot:
            self.fig = plt.figure(figsize=(12,8))
            self.figx = plt.subplot(221)
            self.fig_proc_x = plt.subplot(222)
            self.figy = plt.subplot(223)
            self.fig_proc_y = plt.subplot(224)
            plt.show()

        rospy.spin()

    def receive_imu_data(self, msg):
        """
        For each data point published from the robot:
            1) determine that it is data corresponding to the appropriate step
                and corresponding to physical motion(above threshold value)
            2) append to buffers
            3) if this data point corresponds to near static behavior(below threshold),
                collect last datapoint timestamp, prepare class variables for next step,
                and process data
            TODO: moving avg percent change threshold
        """
        robot_id = int(msg.robot_id)
        try:
            robot_idx = self.robot_idx_map[robot_id]
        except:
            print("robot_id: ", robot_id)
            print("map: ", self.robot_idx_map)
        # ac_num = msg.action_num
        # if ac_num >= 0:
        #     if self.ac_num[robot_idx] == -1 and ac_num == 0 and not self.began[robot_idx]:
        #         # first msg of first step
        #         self.began[robot_idx] = True
        #         self.first_timestamp[robot_idx] = msg.header.stamp.to_sec()
        #         self.ac_num[robot_idx] = ac_num

        #     if ac_num == (self.ac_num[robot_idx] + 1) and self.began[robot_idx]:
        #         # process previous step
        #         last_timestamp = self.latest_timestamp[robot_idx]
        #         self.update_step(robot_idx)
        #         delta_t = last_timestamp - self.first_timestamp[robot_idx]
        #         self.process_data(delta_t, self.ac_num[robot_idx], robot_idx)

        #         # beginning of a new step
        #         print("new step!")
        #         self.ac_num[robot_idx] = ac_num
        #         self.first_timestamp[robot_idx] = msg.header.stamp.to_sec()
        #     # if len(self.ax_buffer[robot_idx]) > 0:
        #     #     print("Action:", ac_num)
        #     #     print("Delta ax:", abs(msg.ax - self.ax_buffer[robot_idx][-1]))
        #     #     print("Delta ay:", abs(msg.ay - self.ay_buffer[robot_idx][-1]))
        #     #     print("Delta gx:", abs(msg.gx - self.gx_buffer[robot_idx][-1]))
        #     #     print("Delta gy:", abs(msg.gy - self.gy_buffer[robot_idx][-1]))

        #     self.ax_buffer[robot_idx].append(100 * msg.ax)
        #     self.ay_buffer[robot_idx].append(100 * msg.ay)
        #     self.gx_buffer[robot_idx].append(RAD_TO_DEG * msg.gx)
        #     self.gy_buffer[robot_idx].append(RAD_TO_DEG * msg.gy)
        #     self.mx_buffer[robot_idx].append(100 * msg.mx)
        #     self.my_buffer[robot_idx].append(100 * msg.my)

        #     self.latest_timestamp[robot_idx] = msg.header.stamp.to_sec()
        print(msg)

            # # arctan2 the magnetometer data to find angle
            # theta = np.arctan2(msg.my, msg.mx)

    def process_data(self, delta_t, ac_num, robot_idx):
        """
        Filter and Transform data, plot if necessary
        """
        print("PROCESSING IMU DATA")
        data = self.data[robot_idx][ac_num]
        ax = data[0]
        ay = data[1]
        gx = data[2]
        gy = data[3]
        mx = data[4]
        my = data[5]

        N = len(ax)

        if N == 0:
            print("No data found this iteration")
            return

        sample_rate = 0

        if delta_t != 0:
            sample_rate = N / delta_t
            print("##############################")
            print("     Processing Step Data")
            print("##############################")
            print(f"Step Time Elapsed: {delta_t} sec")
            print("Number of Samples:", N)
            print("Sample Rate:", sample_rate)

        if self.plot:
            self.figx.cla()
            self.figy.cla()
            self.fig_proc_x.cla()
            self.fig_proc_y.cla()
            print("ax:", ax)
            x = np.arange(N)
            if sample_rate != 0:
                freq = fftfreq(N, 1/sample_rate)[:N//2]
                ##### x-axis data plot #####
                self.figx.plot(x, ax, label="X Acceleration(cm/s)")
                self.figx.plot(x, gx, label="X Angular Velocity(deg/s)")
                self.figx.plot(x, mx, label="X Magnetometer(gauss)")
                self.figx.set_ylim(-100,100)

                ##### x-axis processed plot #####
                self.fig_proc_x.plot(freq, fft(ax)[:N//2], label="X Acceleration Freq")
                self.fig_proc_x.plot(freq, fft(gx)[:N//2], label="X Angular Velocity Freq")

                ##### y-axis data plot #####
                self.figy.plot(x, ay, label="Y Acceleration(cm/s)")
                self.figy.plot(x, gy, label="Y Angular Velocity(deg/s)")
                self.figy.plot(x, my, label="Y Magnetometer(gauss)")
                self.figy.set_ylim(-100,100)

                ##### y-axis processed plot #####
                self.fig_proc_y.plot(freq, fft(ay)[:N//2], label="Y Acceleration Freq")
                self.fig_proc_y.plot(freq, fft(gy)[:N//2], label="Y Angular Velocity Freq")


                self.figx.legend()
                self.figy.legend()
                self.fig_proc_x.legend()
                self.fig_proc_y.legend()

                plt.pause(0.2)
        return

    def de_noise(data, sample_freq):
        # create a butterworth low-pass filter
        cutoff = (1 / len(data)) * 10
        fc = (cutoff * sample_freq / 2)
        wc = 2 * np.pi * cutoff
        # N, Wn = signal.buttord(wp=wc, ws=wc+0.1, gpass=3, gstop=40, analog=False)
        # print(N, Wn)
        sos = signal.butter(4, fc, 'lp', fs=sample_freq, output='sos')
        de_noise_data = signal.sosfilt(sos, data)
        return de_noise_data

    #takes in 1D set of data, eturns a[0], the real-valued DC component of the signal
    def get_dc(data):
        return np.average(np.real(data))

    #takes a 1D set of data, returns a tuple of the (a[0],a[1]), both are real values
    def get_dc_fundamental(data):
        transformed = fourier.rfft(data)
        return (np.real(transformed[0]), np.real(transformed[1]))

    def update_step(self, robot_idx):
        """
        Store action-specific data to the 'data' dict and clear buffers
        """
        # store buffers
        ax = np.array(self.ax_buffer[robot_idx])
        ay = np.array(self.ay_buffer[robot_idx])
        gx = np.array(self.gx_buffer[robot_idx])
        gy = np.array(self.gy_buffer[robot_idx])
        mx = np.array(self.mx_buffer[robot_idx])
        my = np.array(self.my_buffer[robot_idx])

        # log data
        ac_num = self.ac_num[robot_idx]
        self.data[robot_idx][ac_num] = np.vstack((ax, ay, gx, gy, mx, my))

        # reset buffers
        self.ax_buffer[robot_idx]=[]
        self.ay_buffer[robot_idx]=[]
        self.gx_buffer[robot_idx]=[]
        self.gy_buffer[robot_idx]=[]
        self.mx_buffer[robot_idx]=[]
        self.my_buffer[robot_idx]=[]

    def get_step_data(self, robot_idx, ac_num, processed=False):
        """
        Retrieve data corresponding to a specific step
        """
        if processed:
            # TODO: return processed step
            return
        if self.data[robot_idx].get(ac_num) is None:
            print(f"Action {ac_num} exceeds data buffer.")
            return None

        return self.data[robot_idx][ac_num]

if __name__ == "__main__":
    robot_ids = rospy.get_param('/state_publisher/robot_id')
    if not isinstance(robot_ids, list):
        robot_ids = [robot_ids]
    laptop = Laptop(robot_ids, plot=False)
