#!/usr/bin/python3

import rospy
from ros_stuff.msg import RobotCmd, ImuData
from std_msgs.msg import Time
import sys
import board
import adafruit_fxos8700
import adafruit_fxas21002c
i2c = board.I2C()
fxos = adafruit_fxos8700.FXOS8700(i2c)
fxas = adafruit_fxas21002c.FXAS21002C(i2c)

from gpiozero import PWMOutputDevice, DigitalOutputDevice

# below are the correct ones for us
MOTOR_STANDBY = 17 # STBY - 11, GPIO 17
MOTOR_RIGHT_PWM = 18 # PWMB - 12, GPIO 18
MOTOR_RIGHT_FW = 23 # BIN1 - 16, GPIO 23
MOTOR_RIGHT_BW = 24 # BIN2 - 18, GPIO 24
MOTOR_LEFT_PWM = 13 # PWMA - 33, GPIO 13
MOTOR_LEFT_FW = 22 # AIN2 - 15, GPIO 22
MOTOR_LEFT_BW = 27 # AIN1 - 13, GPIO 27
ac_num = -1
def kami_callback(msg):
    print("Hi, I got a message:", msg)

    left_action, right_action, duration, action_num = msg.left_pwm, msg.right_pwm, msg.duration, msg.action_num
    global ac_num
    ac_num = action_num
    if left_action > 0:
        motor_left_forward.on()
        motor_left_backward.off()
    else:
        motor_left_forward.off()
        motor_left_backward.on()

    if right_action > 0:
        motor_right_forward.on()
        motor_right_backward.off()
    else:
        motor_right_forward.off()
        motor_right_backward.on()

    motor_left_pwm.value = abs(left_action)
    motor_right_pwm.value = abs(right_action)

    time_msg = Time()
    time_msg.data = rospy.get_rostime()
    publisher.publish(time_msg)

    print(f"{rospy.get_name()}  ||  L: {left_action}  ||  R: {right_action} || T: {duration}")

    rospy.sleep(duration)

    motor_left_pwm.off()
    motor_right_pwm.off()

def stream_data():
    pub = rospy.Publisher("/imu_data", ImuData, queue_size=1)
    global ac_num
    while not rospy.is_shutdown():
        if ac_num >= 0:
            msg = ImuData()
            msg.a_x, msg.a_y, _ = fxos.accelerometer
            msg.m_x, msg.m_y, _ = fxos.magnetometer
            msg.action_num = ac_num
            print("msg:", msg)
            pub.publish(msg)

if __name__ == '__main__':
    #NEW_SERV_TEST CODE
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

    rospy.init_node(f'robot_{sys.argv[1]}')

    print("waiting for /action_topic rostopic")
    rospy.Subscriber("/action_topic", RobotCmd, kami_callback, queue_size=1)
    print("subscribed to /action_topic")

    publisher = rospy.Publisher("/action_timestamps", Time, queue_size=1)

    #print("rospy spinning")
    #rospy.spin()

    #ROBOT_DATA CODE
    stream_data()
