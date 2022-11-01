#!/usr/bin/python3

from ros_stuff.msg import RobotCmd
import rospy
from ros_stuff.srv import CommandAction, CommandActionResponse  # Service type
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
    print("Hi, I got a message")
    print(request)
    left_forward = request.robot_cmd.left_pwm > 0
    right_forward = request.robot_cmd.right_pwm > 0
    
    if left_forward:
        motor_left_forward.on()
        motor_left_backward.off()
    else:
        motor_left_forward.off()
        motor_left_backward.on()
    motor_left_pwm.on()

    if right_forward:
        motor_right_forward.on()
        motor_right_backward.off()
    else:
        motor_right_forward.off()
        motor_right_backward.on()
    motor_right_pwm.on()

    motor_left_pwm.value = 0.0
    motor_right_pwm.value = 0.0
    
    # Getting requested pwm values
    left_val, right_val, name = request.robot_cmd.left_pwm, request.robot_cmd.right_pwm, request.name
    
    # pub = rospy.Publisher('/{}/cmd_topic'.format(name), String, queue_size=10)

    motor_right_pwm.value = abs(right_val)
    motor_left_pwm.value = abs(left_val)
    action_time = rospy.get_rostime().to_sec()
    print(f"{rospy.get_name()}  ||  left_pwm: {left_val}  ||  right_pwm: {right_val}")
    

    rospy.sleep(0.1)
    motor_left_pwm.value = 0.0
    motor_right_pwm.value = 0.0
    rospy.sleep(0.05)

    return CommandActionResponse(name, action_time)

def robot_server(name):
    # # Create class instance for specific kamigami
    # specific_class = '{}_class'.format(name)
    # specific = KamigamiNode(name)

    # Initialize the server node for specific kamigami
    print("started")
    rospy.init_node('{}_robot_server'.format(name))
    print("made node")
    # Register service
    rospy.Service(
        '/{}/server'.format(name),  # Service name
        CommandAction,  # Service type
        kami_callback  # Service callback
    )
    print("registered service")
    rospy.loginfo('Running robot server...')
    rospy.spin() # Spin the node until Ctrl-C


if __name__ == '__main__':
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
    
    robot_server(sys.argv[1])