from scheduler_func.lat_first_pid import PIDController
import time

lastTime = time.time()


def schedule():
    Kp, Ki, Kd = 1, 0.1, 0.01
    setpoint = 0.8
    dt = time.time() - lastTime
    pidControl = PIDController(Kp, Ki, Kd, setpoint, dt)
    print(f'{pidControl.previous_error}, {pidControl.integral}')


if __name__ == '__main__':
    for i in range(20):
        schedule()

