import numpy as np
import matplotlib.pyplot as plt

#设定神经元参数
#时间参数
T = 100 
dt = 0.1
time = np.arange(0 , T , dt)  #数轴

#神经元核心参数
V_rest = -70  #静息电位
V_th = -60    #阈值电位
V_reset = -70 #重置电位
tau_m = 15    #膜时间常数
I_input = 30  #输入电流
t_ref = 2     #不应期，防止一次受到过大刺激导致通道一直开启导致功耗过大

#随机化数组，模拟随即情况（与simple版的最大区别）
input_spike_rate = 0.8  #输入脉冲发放率,2ms一次
pulse_current = 100      #每次输入脉冲带来的电流增量
#随机的上游神经元发送的脉冲输入
input_spikes = np.random.choice([0, 1] , size=len(time) , p=[1-input_spike_rate * dt , input_spike_rate * dt])
#根据上游神经元的脉冲输入来计算输入电流
I_input = input_spikes * pulse_current

#初始化变量
V = np.ones_like(time) * V_rest  #膜电位数组
spike = np.zeros_like(time)      #脉冲发放数组
ref_count = 0                    #不应期计数器,记录不应期的剩余时间

#模拟神经元活动
for i in range(1 , len(time)):
    #先判断是否处于不应期内，如果是，则直接time++
    if ref_count > 0:  #如果处于不应期
        ref_count -= dt  #减少不应期剩余时间
        V[i] = V_reset   #保持在重置电位
        continue
    
    dV = (-(V[i-1] - V_rest) + I_input[i]) / tau_m * dt  #膜电位变化
    V[i] = V[i-1] + dV  #更新膜电位
    
    if V[i] >= V_th:  #如果达到阈值
        spike[i] = 1   #记录脉冲发放
        V[i] = V_reset #重置膜电位
        ref_count = t_ref / dt #进入不应期

#可视化结果
plt.figure(figsize=(12, 9))

# 子图1：动态输入电流（真实的不确定刺激）
plt.subplot(3, 1, 1)
plt.plot(time, I_input, color='#2ecc71', linewidth=1.2)
plt.ylabel('I_input (nA)', fontsize=12)
plt.title('Single-neuron LIF simulation: Random spike input (realistic brain-like scenario)', fontsize=14, pad=15)
plt.grid(alpha=0.3)

# 子图2：神经元膜电位变化
plt.subplot(3, 1, 2)
plt.plot(time, V, color='#3498db', label='膜电位', linewidth=1.2)
plt.axhline(V_th, color='#e74c3c', linestyle='--', label='阈值电位', linewidth=1.5)
plt.ylabel('V_th (mV)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)

# 子图3：神经元输出脉冲
plt.subplot(3, 1, 3)
plt.plot(time, spike, color='#2c3e50', linewidth=1.2)
plt.ylabel('spike', fontsize=12)
plt.xlabel('time (ms)', fontsize=12)
plt.ylim(-0.1, 1.1)  # 让脉冲显示更清晰
plt.grid(alpha=0.3)

plt.tight_layout()  # 自动调整子图间距
plt.show()