from __future__ import annotations
from time import sleep
from typing import TYPE_CHECKING
from gpiozero import Motor, Servo
import board
from pathlib import Path
from keyboard import wait

from src.tof import start_TOF
from src.process import EmbeddingProcessor
from src.capture import capture_still
from garment_db import GarmentDB


# for type hint
if TYPE_CHECKING:
    from adafruit_vl53l0x import VL53L0X
    from microcontroller import Pin
    import numpy as np

MOTOR_FORWARD_PIN = 0
MOTOR_BACKWARD_PIN = 1
MOTOR_SPEED = 1

# 센서 하나가 동작하지 않는 관계로 TOF 센서는 하나만 사용할 예정
#TOF_START_XSHUT_PIN : Pin = board.D21
#TOF_FINISH_XSHUT_PIN : Pin = board.D20
TOF_DETECT_DISTANCE_THRESHOLD = 100 # 가까이 있다고 인식하는 거리
TOF_DETECT_INTERVAL = 0.1 # 센서 주기
TOF_DETECT_TIME_THRESHOLD = 0.5 # 몇 초 동안 가까이 있어야 할지

SERVO_PIN = 18

DEFAULT_MODEL = Path('models/mobilenetv3_small_embedding.onnx')

DB = Path('garments.db')

THRESHOLD = 0.7

CATEGORY = {1 : '유색 외출복',
            2 : '무색 외출복',
            3 : '내복'}

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

def _print_search_result(result : dict, end : None = None):
    '''print a result of db search'''
    print(f"\tid={result['garment_id']}, name={result['name']}, "
          f"category={result['category']}, similarity={result['similarity']:.4f}",
          end=end)

def _add_garment_to_db(name : str,
                 category : str,
                 embedding : np.ndarray,
                 db : GarmentDB):
    garment_id = db.add_garment(name, category)
    print(f"의류 등록: id={garment_id}, name={name}, category={category}")

    embedding_id = db.add_embedding(garment_id, embedding)
    print(f"임베딩 저장 완료: garment_id={garment_id}, embedding_id={embedding_id}, dimension={embedding.size}")
   

def main() -> None:
    model = EmbeddingProcessor(DEFAULT_MODEL)

    motor = Motor(
        forward=MOTOR_FORWARD_PIN,
        backward=MOTOR_BACKWARD_PIN,
        pwm=True
        )

    # sensor_start, sensor_finish = start_TOF(TOF_START_XSHUT_PIN, TOF_FINISH_XSHUT_PIN)
    sensor_finish = start_TOF()
    
    db = GarmentDB(DB)

    servo = Servo(SERVO_PIN,
                  min_pulse_width=0.5 / 1000,
                  max_pulse_width=2.5 / 1000)
    
    try:
        while True:
            # 스페이스 바가 눌릴 때까지 기다리기
            wait('space')

            motor.forward(MOTOR_SPEED)
            tof_detect(sensor_finish)
            motor.stop()

            img_rgb = capture_still(camera_num=0, size=(640, 480))
            embedding = model.create_embedding(img_rgb)

            matches = db.find_similar(embedding, top_k=5)
            print('유사도 검색 결과:')
            
            for result in matches:
                _print_search_result(result=result)
            
            matched = [result for result in matches if result['similarity'] >= THRESHOLD]
            if not matched:
                print(f'임계값 {THRESHOLD} 이상의 의류가 없습니다.')
                print('새 의류를 등록하시겠습니까? [Y/N] ')
                match input():
                    case 'Y' | 'y':
                        name = input('의류 이름을 입력해 주십시오 >>\n')
                        category = CATEGORY[int(
                            input('''어느 것으로 분류할 것인지 입력해 주십시오
                                유색 외출복 : 1, 무색 외출복 : 2, 내복 : 3
                                >>'''))]
                        
                        _add_garment_to_db(name, category, embedding, db)
                        
                    case _:
                        print('''새 의류를 등록하지 않겠습니다.
                            임시로 유색 외복에 분류합니다.''')
                        category = '유색 외복'
                        name = '등록되지 않은 의류'
                
            else:
                best = matched[0]
                category = best['category']
                name = best['name']
            print(f'{name}: {category}로 분류합니다.')
            
            match category:
                case '유색 외복':
                    servo.max()
                case '무색 외복':
                    servo.mid()
                case '내복':
                    servo.min()
            

    except KeyboardInterrupt:
        print('\nEnd the program.')

if __name__ == '__main__':
    main()