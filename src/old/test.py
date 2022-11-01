#!/usr/bin/python3
import rospy
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

def talker():
    # setup
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
    # main code 
    pub = rospy.Publisher('/chatter_talk', String, queue_size=10)
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        for i in range(100):
            right = 0.5 * abs(math.cos(i))
            left = 0.55 * abs(math.cos(i - math.pi / 2))
            motor_right_pwm.value = right
            motor_left_pwm.value = left
            pubstr = "kami pwm " + str(left) + ", " + str(right)
            pub.publish(pubstr)
            print(rospy.get_name() + ": I sent", pubstr)
            r.sleep()
        r.sleep()
    motor_left_pwm.value = 0.0
    motor_right_pwm.value = 0.0
    for port in ports:
        port.off()
    print("done")

if __name__ == '__main__':
    rospy.init_node('talker', anonymous=True)
    try:
        talker()
    except rospy.ROSInterruptException: pass
