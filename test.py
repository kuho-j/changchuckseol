import time
from src.tfo import start_TFO

if __name__ == '__main__':
    sensor = start_TFO()
    
    try:
        while True:
            distance = sensor.range
            print(f'distance : {distance}mm')
            time.sleep(0.5)

    except KeyboardInterrupt:
        print('quit')