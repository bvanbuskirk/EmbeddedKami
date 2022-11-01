#!/usr/bin/python3

from ros_stuff.msg import RobotCmd
import numpy as np
import rospy
from ros_stuff.srv import CommandAction  # Service type
import sys
from std_msgs.msg import String
import math

from gpiozero import PWMOutputDevice, DigitalOutputDevice

# below are the correct ones for us
MOTOR_STANDBY = 17 # STBY - 11, GPIO 17
MOTOR_RIGHT_PWM = 18 # PWMB - 12, GPIO 18
MOTOR_RIGHT_FW = 23 # BIN1 - 16, GPIO 23
MOTOR_RIGHT_BW = 24 # BIN2 - 18, GPIO 24
MOTOR_LEFT_PWM = 13 # PWMA - 33, GPIO 13
MOTOR_LEFT_FW = 22 # AIN2 - 15, GPIO 22
MOTOR_LEFT_BW = 27 # AIN1 - 13, GPIO 27


def kami_callback(request):
    motor_standby = DigitalOutputDevice(MOTOR_STANDBY)
    motor_left_pwm = PWMOutputDevice(MOTOR_LEFT_PWM)
    motor_left_forward = DigitalOutputDevice(MOTOR_LEFT_FW)
    motor_left_backward = DigitalOutputDevice(MOTOR_LEFT_BW)
    motor_right_pwm = PWMOutputDevice(MOTOR_RIGHT_PWM)
    motor_right_forward = DigitalOutputDevice(MOTOR_RIGHT_FW)
    motor_right_backward = DigitalOutputDevice(MOTOR_RIGHT_BW)
    ports = [motor_standby, motor_left_pwm, motor_left_forward, motor_left_backward,
        motor_right_pwm, motor_right_forward, motor_right_backward]
    motor_standby.on()
    
    motor_left_forward.on()
    motor_left_backward.off()
    motor_left_pwm.on()

    motor_right_forward.on()
    motor_right_backward.off()
    motor_right_pwm.on()

    motor_left_pwm.value = 0.0
    motor_right_pwm.value = 0.0
    
    #Getting requested pwm values
    left_val, right_val, name = request.robot_cmd.left_pwm, request.robot_cmd.right_pwm, request.name

    pub = rospy.Publisher('/{}/cmd_topic'.format(name), String, queue_size=10)

    r = rospy.Rate(10)

    while not rospy.is_shutdown():
        # class_name.set_pwm(left_val, right_val)
        motor_right_pwm.value = right_val
        motor_left_pwm.value = left_val
        pubstr = name + " pwm " + str(left_val) + ", " + str(right_val)
        pub.publish(pubstr)
        print(rospy.get_name() + ": I sent", pubstr)
        pub.publish(pubstr)  # Publish to cmd_topic
        r.sleep()

def robot_server(name):
    # Initialize the server node for specific kamigami
    rospy.init_node('{}_robot_server'.format(name))
    # Register service
    rospy.Service(
        '/{}/server'.format(name),  # Service name
        CommandAction,  # Service type
        kami_callback  # Service callback
    )
    rospy.loginfo('Running robot server...')
    rospy.spin() # Spin the node until Ctrl-C


if __name__ == '__main__':
    robot_server(sys.argv[1])