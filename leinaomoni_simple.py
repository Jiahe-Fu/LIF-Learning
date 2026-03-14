import numpy
import matplotlib.pyplot as plt

#设定神经元参数
T = 100
dt = 0.1
time = numpy.arange(0, T, dt)

#神经元核心参数
V_rest = -70
V_th = -55
V_reset = -70
tau_m = 10
I_input = 30
#初始化变量
V = numpy.ones_like(time) * V_rest
spike = numpy.zeros_like(time)

#模拟神经元活动
for i in range(1, len(time)):
    dV = (-(V[i-1] - V_rest) + I_input) / tau_m * dt
    V[i] = V[i-1] + dV
    
    if V[i] >= V_th:
        spike[i] = 1
        V[i] = V_reset
#可视化结果
plt.figure(figsize=(10, 6))
# 上图：膜电位变化
plt.subplot(2,1,1)
plt.plot(time, V, label='膜电位')
plt.axhline(V_th, color='r', linestyle='--', label='阈值电位')
plt.ylabel('膜电位 (mV)')
plt.title('你的第一个LIF神经元仿真结果')
plt.legend()

# 下图：脉冲发放
plt.subplot(2,1,2)
plt.plot(time, spike, color='k')
plt.ylabel('脉冲发放')
plt.xlabel('时间 (ms)')
plt.ylim(-0.1, 1.1)  # 让脉冲看得更清楚
plt.show()