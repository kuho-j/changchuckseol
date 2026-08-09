from gpiozero import OutputDevice

# IN1, IN2 핀 설정 (디지털 출력)
in1 = OutputDevice(22)
in2 = OutputDevice(27)

def forward():
    """최대 속도 정방향 회전"""
    in1.on()
    in2.off()

def backward():
    """최대 속도 역방향 회전"""
    in1.off()
    in2.on()

def stop():
    """모터 정지"""
    in1.off()
    in2.off()

try:
    forward()
except KeyboardInterrupt:
    stop()
    print("\n프로그램 종료")