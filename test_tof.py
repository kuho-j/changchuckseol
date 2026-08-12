'''
tfo 센서 테스트
'''
import time
from tof import start_TOF

if __name__ == '__main__':
    sensor = start_TOF()
    
    try:
        while True:
            distance = sensor.range
            print(f'distance : {distance}mm')
            time.sleep(0.5)

    except KeyboardInterrupt:
        print('quit')