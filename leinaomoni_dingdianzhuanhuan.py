import numpy as np
import matplotlib.pyplot as plt
# 设定定点的参数
SCALE = 1 << 8 # 定点数的缩放影子，要用二进制下好表示的，所以不能用1000
DATA_TYPE = int
#所有数据全部int(num * SCALE)来表示，输出的时候再除以SCALE
# 设定神经元参数
T = 100
dt = 0.1    #时间只是记录循环次数，不用定点表示了，毕竟是循环次数，verilog时是没有的
time = np.arange(0, T, dt)  # 数轴
# 神经元核心参数
V_rest = int (-70 * SCALE)  # 静息电位    
V_th = int(-60 * SCALE)     # 阈值电位
V_reset = int (-70 * SCALE) # 重置电位
tau_m = 15                  # 膜时间常数，同下
t_ref = 2                   # 不应期，防止一次受到过大刺激导致通道一直开启导致功耗过大，这里也是时间量，不用定点表示
# 这里没有输入电流，上面也搞错了，输入电流是随机计算的，但是问题不大，因为会覆盖掉

# 随机化输入
input_spike_rate = 0.8  # 输入脉冲发放率,1ms发0.8个
pulse_current = int (120 * SCALE)      # 每次输入脉冲带来的电流增量
# 随机的上游神经元发送的脉冲输入
input_spikes = np.random.choice([0, 1], size=len(time), p=[1 - input_spike_rate * dt, input_spike_rate * dt])
# 根据上游神经元的脉冲输入来计算输入电流
I_input = input_spikes * pulse_current

#初始化变量
V = np.ones_like(time, dtype=DATA_TYPE) * V_rest    # 膜电位数组, 这里V_rest已经是定点表示了，所以V也是定点表示的，不必再乘256
spike = np.zeros_like(time, dtype=DATA_TYPE)        # 脉冲发放数组，这里只是记录spike是否发送，所以不需要定点表示
ref_count = 0                                       # 不应期计数器,记录不应期的剩余时间，这里也是时间量，不用定点表示

#搞一个定点的时间，方便后续计算
dt_fixed = int (dt * SCALE) # 定点表示的dt，方便后续计算
tau_m_fixed = int (tau_m * SCALE) # 定点表示的tau_m，方便后续计算


# 模拟神经元活动
for i in range(1, len(time)):
    if ref_count > 0:    # 如果处于不应期
        ref_count -= dt  # 减少不应期剩余时间
        V[i] = V_reset   # 保持在重置电位
        continue
    else:
        # 这里公式需要改动，改成定点数的乘法
        # 原公式: dV = (-(V[i - 1] - V_rest) + I_input[i]) / tau_m * dt
        dV_fixed = (-(V[i - 1] - V_rest) + I_input[i]) * dt_fixed // tau_m_fixed # 定点数的乘法, 这里乘了一个定点数又除了一个定点数所以还是定点数
        V[i] = V[i - 1] + dV_fixed                                               # 更新膜电位，这里是定点数的加法，结果还是定点数
        
        if V[i] >= V_th:   # 如果达到阈值
            spike[i] = 1   # 记录脉冲发放
            V[i] = V_reset # 重置膜电位
            ref_count = int(t_ref / dt) # 进入不应期
# 可视化结果
# 定点数转浮点数（用于画图，和原浮点数模型对比）
V_float = V / SCALE
I_input_float = I_input / SCALE
V_th_float = V_th / SCALE

# 画图
plt.figure(figsize=(12, 9))

# 子图1：输入电流
plt.subplot(3,1,1)
plt.plot(time, I_input_float, color='#2ecc71')
plt.ylabel('Input Current (nA)')
plt.title('Fixed-point LIF Neuron Simulation (Q8.8)', fontsize=14)
plt.grid(alpha=0.3)

# 子图2：膜电位
plt.subplot(3,1,2)
plt.plot(time, V_float, color='#3498db', label='Membrane Potential')
plt.axhline(V_th_float, color='#e74c3c', linestyle='--', label='Threshold (-60mV)')
plt.ylabel('Membrane Potential (mV)')
plt.legend()
plt.grid(alpha=0.3)

# 子图3：输出脉冲
plt.subplot(3,1,3)
plt.plot(time, spike, color='#2c3e50')
plt.ylabel('Output Spike')
plt.xlabel('Time (ms)')
plt.ylim(-0.1, 1.1)
plt.grid(alpha=0.3)

plt.tight_layout()  # 自动调整子图间距
plt.show()