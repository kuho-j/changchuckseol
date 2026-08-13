from gpiozero import Servo
from time import sleep

SERVO_PIN = 18

if __name__ == '__main__':
    servo = Servo(SERVO_PIN, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

    try:
        while True:
            servo.mid()
            sleep(1)
            
            servo.min()
            sleep(1)
            
            servo.max()
            sleep(1)
            
    except KeyboardInterrupt:
        print('\nEnd the program')
        servo.detach()