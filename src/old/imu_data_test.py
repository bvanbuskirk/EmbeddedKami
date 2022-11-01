#!/usr/bin/python3
import matplotlib.pyplot as plt
from matplotlib import animation as anim, cm
import rospy
import numpy as np
import collections

# from std_msgs.msg import float64
# from ros_stuff.srv import CommandAction
from ros_stuff.msg import ImuData

class Laptop():
    def __init__(self):
        rospy.init_node("laptop")
        rospy.Subscriber("/imu_data", ImuData, self.callback, queue_size=1)
        print("initialized")
        
        self.fig = plt.figure(figsize=(12,8))
        self.figx = plt.subplot(311)
        self.figy = plt.subplot(312)
        self.figz = plt.subplot(313)

        self.aX, self.aY, self.aZ = collections.deque(np.zeros(10)), collections.deque(np.zeros(10)), collections.deque(np.zeros(10))
        self.gX, self.gY, self.gZ = collections.deque(np.zeros(10)), collections.deque(np.zeros(10)), collections.deque(np.zeros(10))
        plt.show()
        rospy.spin()

    def callback(self, msg):
        # update deques
        self.aX.popleft()
        self.aX.append(msg.a_x)
        self.aY.popleft()
        self.aY.append(msg.a_y)
        self.aZ.popleft()
        self.aZ.append(msg.a_z)

        self.gX.popleft()
        self.gX.append(msg.g_x)
        self.gY.popleft()
        self.gY.append(msg.g_y)
        self.gZ.popleft()
        self.gZ.append(msg.g_z)

        # clear axes
        self.figx.cla()
        self.figy.cla()
        self.figz.cla()

        # === plot X ===
        self.figx.plot(self.aX)
        self.figx.scatter(len(self.aX)-1, self.aX[-1])
        self.figx.text(len(self.aX)-1, self.aX[-1]+2, "{}%".format(self.aX[-1]))

        self.figx.plot(self.gX)
        self.figx.scatter(len(self.gX)-1, self.gX[-1])
        self.figx.text(len(self.gX)-1, self.gX[-1]+2, "{}%".format(self.gX[-1]))

        self.figx.set_ylim(-25,25)

        # === plot Y ===
        self.figy.plot(self.aY)
        self.figy.scatter(len(self.aY)-1, self.aY[-1])
        self.figy.text(len(self.aY)-1, self.aY[-1]+2, "{}%".format(self.aY[-1]))

        self.figy.plot(self.gY)
        self.figy.scatter(len(self.gY)-1, self.gY[-1])
        self.figy.text(len(self.gY)-1, self.gY[-1]+2, "{}%".format(self.gY[-1]))

        self.figy.set_ylim(-25,25)

        # === plot Z ===
        self.figz.plot(self.aZ)
        self.figz.scatter(len(self.aZ)-1, self.aZ[-1])
        self.figz.text(len(self.aZ)-1, self.aZ[-1]+2, "{}%".format(self.aZ[-1]))

        self.figz.plot(self.gZ)
        self.figz.scatter(len(self.gZ)-1, self.gZ[-1])
        self.figz.text(len(self.gZ)-1, self.gZ[-1]+2, "{}%".format(self.gZ[-1]))

        self.figz.set_ylim(-25,25)

        plt.pause(0.001)


if __name__ == "__main__":
    laptop = Laptop()

    # proxy = rospy.ServiceProxy("/kamigami", CommandAction)

    # while not rospy.is_shutdown():
        # logic to get action
        # action = ...

        # cmd = RobotCmd()
        # cmd.left_pwm = action[0]
        # cmd.right_pwm = action[1]
        # cmd.duration = action[2]

        # timestamp = proxy(cmd)
