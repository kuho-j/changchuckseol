from __future__ import annotations
from time import sleep
from typing import TYPE_CHECKING
from gpiozero import Motor
import board

from tof import start_TOF


# for type hint
if TYPE_CHECKING:
    from adafruit_vl53l0x import VL53L0X
    from microcontroller import Pin

MOTOR_FORWARD_PIN = 0
MOTOR_BACKWARD_PIN = 1
MOTOR_SPEED = 0.5

# 센서 하나가 동작하지 않는 관계로 TOF 센서는 하나만 사용할 예정
TOF_START_XSHUT_PIN : Pin = board.D21
TOF_FINISH_XSHUT_PIN : Pin = board.D20
TOF_DETECT_DISTANCE_THRESHOLD = 100 # 가까이 있다고 인식하는 거리
TOF_DETECT_INTERVAL = 0.1 # 센서 주기
TOF_DETECT_TIME_THRESHOLD = 0.5 # 몇 초 동안 가까이 있어야 할지

def tof_detect(tof : VL53L0X):
    '''
    tof에 물체가 감지되면 1을 리턴
    '''

    while True:
        distance = tof.range
        if distance < TOF_DETECT_DISTANCE_THRESHOLD:
            for _ in range(int(TOF_DETECT_TIME_THRESHOLD // TOF_DETECT_INTERVAL)):
                if distance > TOF_DETECT_DISTANCE_THRESHOLD:
                    continue
                sleep(TOF_DETECT_INTERVAL)

            return 1
        sleep(TOF_DETECT_INTERVAL)
    
'''
def test_tof():
    try:
        while True:
            if tof_detect(sensor_start):
                print('detect a thing!')
    except KeyboardInterrupt:
        print('end the program')
'''

if __name__ == '__main__':
    motor = Motor(
        forward=MOTOR_FORWARD_PIN,
        backward=MOTOR_BACKWARD_PIN
        )

    # sensor_start, sensor_finish = start_TOF(TOF_START_XSHUT_PIN, TOF_FINISH_XSHUT_PIN)
    sensor_start = start_TOF()
    
    tof_detect(sensor_start)
    motor.forward(MOTOR_SPEED)
    
    