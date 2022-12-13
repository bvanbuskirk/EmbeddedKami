#!/usr/bin/python3

import rospy
from ros_stuff.msg import RobotCmd, ImuData

from std_msgs.msg import Time, Header
import numpy as np
import pickle as pkl
import sys
import os
import time
import busio
import digitalio
import board
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn
import adafruit_fxos8700
import adafruit_fxas21002c
from scipy.signal import savgol_filter
from scipy import integrate
i2c = board.I2C()
fxos = adafruit_fxos8700.FXOS8700(i2c)
fxas = adafruit_fxas21002c.FXAS21002C(i2c)

from gpiozero import PWMOutputDevice, DigitalOutputDevice
# from gpiozero import Motor, DigitalOutputDevice

# below are the correct ones for us
MOTOR_STANDBY = 17 # STBY - 11, GPIO 17
MOTOR_RIGHT_PWM = 18 # PWMB - 12, GPIO 18
MOTOR_RIGHT_FW = 23 # BIN1 - 16, GPIO 23
MOTOR_RIGHT_BW = 24 # BIN2 - 18, GPIO 24
MOTOR_LEFT_PWM = 13 # PWMA - 33, GPIO 13
MOTOR_LEFT_FW = 22 # AIN2 - 15, GPIO 22
MOTOR_LEFT_BW = 27 # AIN1 - 13, GPIO 27

#global vars
ac_num = -1
count = 0
running_sum = 0.0
start_sampling = False
la = 0.0
ra = 0.0

def kami_callback(msg):
    print("\n\nHi, I got a message:", msg)
    global start_sampling
    global ac_num
    global la
    global ra
    start_sampling = True
    left_action, right_action, duration, action_num = msg.left_pwm, msg.right_pwm, msg.duration, msg.action_num
    ac_num = action_num
    la = left_action
    ra = right_action
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
    start_sampling = False
    measure_battery()

def measure_battery():
    global count
    global running_sum
    inv_volt_divider = 2.31
    running_sum += battery.voltage * inv_volt_divider
    count += 1
    if count >= 5:
        batt_voltage = running_sum / count
        running_sum = 0.0
        count = 0
        print("\nCURRENT BATTERY VOLTAGE: ", batt_voltage)
        print("CHARGE WHEN BELOW 3.9V\n")


def stream_data(robot_id):
    print("sampling data")
    global start_sampling
    global ac_num
    global la
    global ra

    sample_idx = 0
    step_idx = 0
    bias_idx = 0

    """
    data shape: 120 steps x 10 sensor axes x 80 buffer length
    axes:   0 - ts
            1 - ax
            2 - ay
            3 - az
            4 - gx
            5 - gy
            6 - gz
            7 - mx
            8 - my
            9 - mz
            10 - left PWM
            11 - right PWM
    """
    data = np.zeros((120, 12, 80))

    # Biases
    ax_bias = 0
    ay_bias = 0
    az_bias = 0
    ax_bias_buff = np.empty(20)
    ay_bias_buff = np.empty(20)
    ay_bias_buff = np.empty(20)

    gx_bias = 0
    gy_bias = 0
    gz_bias = 0
    gx_bias_buff = np.empty(20)
    gy_bias_buff = np.empty(20)
    gy_bias_buff = np.empty(20)

    mx_bias = 0
    my_bias = 0
    mz_bias = 0
    mx_bias_buff = np.empty(20)
    my_bias_buff = np.empty(20)
    my_bias_buff = np.empty(20)

    while not rospy.is_shutdown():
        ax, ay, az = fxos.accelerometer
        gx, gy, gz = fxas.gyroscope
        mx, my, mz = fxos.magnetometer

        if step_idx == 120:
            with open("149_imu_data.pkl", "wb") as f:
                pkl.dump(data, f)


        if start_sampling:
            # store sampled data
            data[step_idx, :, sample_idx] = np.array([time.time(), ax, ay, az, gx, gy, gz, mx, my, mz, la, ra])
            sample_idx += 1

        elif sample_idx != 0:
            # process buffers - unbias, filter and integrate
            # t1 = time.time()
            print(f"Processing {sample_idx} samples!")

            # isolate and unbias
            bias = np.array([data[step_idx, 0, 0], ax_bias, ay_bias, az_bias, gx_bias, gy_bias, gz_bias, mx_bias, my_bias, mz_bias]).reshape((10,1))
            data[step_idx, :10, :sample_idx - 1] -= bias
            # # filter
            # window1 = int(idx * 0.25)
            # window2 = int(idx * 0.75)
            # window1 = window1 + 1 if window1 % 2 == 0 else window1
            # window2 = window2 + 1 if window2 % 2 == 0 else window2
            # gx_filt_25 = savgol_filter(gx_raw, window1, 3)
            # gy_filt_25 = savgol_filter(gy_raw, window1, 3)

            # gx_filt_75 = savgol_filter(gx_raw, window2, 3)
            # gy_filt_75 = savgol_filter(gy_raw, window2, 3)
            # print("\n\nPitch_raw: ", gy_raw)

            # print("\nGx MSE 25% window: ", (gx_raw - gx_filt_25).mean())
            # print("Gy MSE 25% window: ", (gy_raw - gy_filt_25).mean())
            # # print("Gy_filt_25: ", gy_filt_25)

            # print("\nGx MSE 75% window: ", (gx_raw - gx_filt_75).mean())
            # print("Gx MSE 75% window: ", (gy_raw - gy_filt_75).mean())
            # # print("Gy_filt_75: ", gy_filt_75)

            # print("\nRaw Int: ", np.trapz(gy_raw, ts_sample))
            # print("Bool Sum: ", np.sum(gy_raw < 0.0) / idx)
            # print("25% Int: ", np.trapz(gy_filt_25, ts_sample))
            # print("75% Int: ", np.trapz(gy_filt_75, ts_sample))


            # # integrate
            # roll = np.trapz(gx_filt_75, ts_sample)
            # pitch = np.trapz(gy_filt_75, ts_sample)





            # determine heading
            sample_idx = 0
            step_idx += 1
            # print(f"Processing Delay: {time.time() - t1} secs")
            # send data
            # msg = ImuData()
            # msg.vx = np.mean(roll)
            # msg.vy = np.mean(pitch)
            # msg.theta = theta
            # msg.header = Header()
            # msg.header.stamp = rospy.Time.now()
            # msg.robot_id = robot_id
            # pub.publish(msg)
            # print(f"Processing + Publishing Delay: {time.time() - t1} secs")

        else:
            # update bias
            if bias_idx >= len(gx_bias_buff):
                bias_idx = 0
                gx_bias = np.mean(gx_bias_buff)
                gy_bias = np.mean(gy_bias_buff)
                # print(f"X Gyro Bias: {gx_bias}    Y Gyro Bias: {gy_bias}")
            # store data for next update
            gx_bias_buff[bias_idx] = gx
            gy_bias_buff[bias_idx] = gy
            bias_idx += 1

        rospy.sleep(0.001)

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
    spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
    cs = digitalio.DigitalInOut(board.D16)
    adc = MCP.MCP3008(spi, cs)
    battery = AnalogIn(adc, MCP.P0)

    robot_name = sys.argv[1]
    robot_id = int(robot_name[-1])
    rospy.init_node(f'robot_{robot_name}')

    print("waiting for /action_topic rostopic")
    rospy.Subscriber("/action_topic", RobotCmd, kami_callback, queue_size=1)
    print("subscribed to /action_topic")

    publisher = rospy.Publisher("/action_timestamps", Time, queue_size=1)
    pub = rospy.Publisher("/imu_data", ImuData, queue_size=1)

    print("rospy spinning")
    # rospy.spin()
    stream_data(robot_id)
