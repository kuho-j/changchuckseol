from gpiozero import Motor

if __name__ == '__main__':
    motor = Motor(22, 27, pwm=False)
    try:
        motor.forward(1)
    except KeyboardInterrupt:
        motor.stop()
