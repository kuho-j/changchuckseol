from gpiozero import Motor
from time import sleep

if __name__ == '__main__':
    motor = Motor(22, 27, pwm=False)
    try:
        while True:
            motor.forward(1)
            sleep(1)
    except KeyboardInterrupt:
        motor.stop()
