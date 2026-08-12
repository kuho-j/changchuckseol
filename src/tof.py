'''
TFO 센서 VL53L0를 위한 파일
센서 한 개를 사용할 때와, 두 개를 사용할 때 모두 지원하도록 만들기

사용 예시

sensor_1, sensor_2 = start_TOF(..., ...)
sensor = start_TOF(...)
'''

import time
import board
import busio
from microcontroller import Pin
from digitalio import DigitalInOut
from adafruit_vl53l0x import VL53L0X

def start_TOF(
    xshut_pin_0 : Pin | None = None,
    xshut_pin_1 : Pin | None = None
    ) -> VL53L0X | tuple[VL53L0X, VL53L0X]:

    i2c = busio.I2C(board.SCL, board.SDA)

    # if there is one sensor
    if xshut_pin_1 is None:
        return VL53L0X(i2c, address=0x29)
    
    # if there are two sensors
    xshut_0 = DigitalInOut(xshut_pin_0)
    xshut_1 = DigitalInOut(xshut_pin_1)

    xshut_0.switch_to_output(value=False)
    xshut_1.switch_to_output(value=False)
    time.sleep(0.1)
    
    xshut_1.value = True
    time.sleep(0.05)
    
    sensor_1 = VL53L0X(i2c)
    sensor_1.set_address(0x30)

    xshut_0.value = True
    time.sleep(0.05)
    
    sensor_0 = VL53L0X(i2c, address=0x29)

    return sensor_0, sensor_1
        
