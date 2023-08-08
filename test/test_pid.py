import time
import matplotlib.pyplot as plt
import numpy as np
# from scipy.interpolate import spline
from scipy.interpolate import BSpline, make_interp_spline  # Switched to BSpline

import time


class PID:
    """PID Controller
    """

    def __init__(self, P=0.2, I=0.0, D=0.0, current_time=None):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_interval = 0
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

        self.PTerm_list = []
        self.ITerm_list = []
        self.DTerm_list = []

        """Clears PID computations and coefficients"""
        self.targetPoint = 0

        self.PTerm = 0
        self.ITerm = 0
        self.DTerm = 0
        self.last_error = 0

        # Windup Guard
        # 积分误差上限
        self.windup_guard = 20

        self.output = 0

    def update(self, feedback_value, current_time=None):
        """Calculates PID value for given reference feedback

        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}

        .. figure:: images/pid_1.png
           :align:   center

           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)

        """
        error = self.targetPoint - feedback_value

        self.current_time = current_time if current_time is not None else time.time()
        delta_time = self.current_time - self.last_time
        delta_error = error - self.last_error

        if delta_time >= self.sample_interval:
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if self.ITerm < -self.windup_guard:
                self.ITerm = -self.windup_guard
            elif self.ITerm > self.windup_guard:
                self.ITerm = self.windup_guard

            self.DTerm = 0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error = error

            self.output = self.PTerm + self.Ki * self.ITerm + self.Kd * self.DTerm
        self.PTerm_list.append(self.PTerm)
        self.ITerm_list.append(self.Ki * self.ITerm)
        self.DTerm_list.append(self.Kd * self.DTerm)
        return self.output

    def get_values(self):
        return self.PTerm_list, self.ITerm_list, self.DTerm_list

    def set_target(self, target_value):
        self.targetPoint = target_value

    def set_kp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def set_ki(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def set_kd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def set_windup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

    def set_sample_interval(self, sample_interval):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_interval = sample_interval



np.random.seed(114514)


# 模拟电机，可以设置角速度
class Motor:
    def __init__(self, angle=0, angular_velocity=0):
        # 角速度控制在[-pi,pi]
        # 角度范围[-pi,pi]
        self.angle = angle
        self.angular_velocity = angular_velocity

    def update(self, step=1):
        if self.angular_velocity >= np.pi:
            self.angular_velocity = np.pi
        elif self.angular_velocity <= -np.pi:
            self.angular_velocity = -np.pi
        # 添加了点随机扰动
        self.angle += self.angular_velocity * step + np.random.uniform(-np.pi / 100, np.pi / 100)
        if self.angle >= 0:
            self.angle = self.angle % np.pi
        else:
            self.angle = -(np.abs(self.angle) % np.pi)


def test_pid(P=0.2, I=0.0, D=0.0, L=100):
    pid_controller = PID(P, I, D, current_time=0)
    # 设置目标值
    pid_controller.set_target(np.pi / 2)
    # 设置采样间隔
    pid_controller.set_sample_interval(1)
    motor = Motor()
    END = L
    # 电机当前角度
    feedback = motor.angle

    feedback_list = []
    time_list = []
    target_point_list = []
    output_list = []

    for i in range(1, END):
        # pid_controller.set_target(np.pi * np.sin(i / 10))
        # 获取PID控制器输出的角速度
        output = pid_controller.update(feedback, current_time=i)
        # 将角速度作用于电机
        motor.angular_velocity = output
        # 按照当前设置的角速度更新电机角度
        motor.update()
        # 获取更新后电机的新角度
        feedback = motor.angle
        # time.sleep(0.02)
        feedback_list.append(feedback)
        output_list.append(output)
        target_point_list.append(pid_controller.targetPoint)
        time_list.append(i)

    # 存储一些中间变量方便查看
    PTerm_list, ITerm_list, DTerm_list = pid_controller.get_values()
    time_sm = np.array(time_list)
    # 插值到300个点，方便绘图展示
    time_smooth = np.linspace(np.min(time_sm), np.max(time_sm), 300)

    # 根据time_smooth 给中间变量也进行插值
    feedback_smooth = make_interp_spline(time_list, feedback_list)(time_smooth)
    PTerm_smooth = make_interp_spline(time_list, PTerm_list)(time_smooth)
    ITerm_smooth = make_interp_spline(time_list, ITerm_list)(time_smooth)
    DTerm_smooth = make_interp_spline(time_list, DTerm_list)(time_smooth)
    output_smooth = make_interp_spline(time_list, output_list)(time_smooth)

    # 选择角度变量进行绘图
    plt.plot(time_smooth, feedback_smooth)
    plt.plot(time_list, target_point_list)
    plt.xlim((0, L))
    plt.xlabel('time (s)')
    plt.ylabel('PID (PV)')
    plt.title('TEST PID')

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test_pid(0.3, 0.001, 0.001, L=500)
    test_pid(L=500)

