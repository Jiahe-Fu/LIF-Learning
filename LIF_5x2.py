# 这一份是完成一个5输入，2输出的脉冲神经网络的算法验证
# 这里的核心是本来一对一输入输出只用I = w * s，这里的w代表突触强度。s代表是否有脉冲，现在w是n_input * n_output的矩阵，s是大小为n_input的向量
import numpy as np
import matplotlib.pyplot as plt
# from brian2 import *

# 设定参数
n_input = 5     # 5输入
n_output = 2    # 2输出
# 神经参数
V_th = -65      # 阈值电位为-60
V_reset = -70   # 重置电位为-70，即到达-60后重置回-70
V_rest = -70    # 静息电位为-70
tau_m = 30      # 膜时间常数15ms,即指数衰减下每15ms衰减一半，即半衰期为15ms
t_ref = 2       # 不应期2ms
# STDP学习参数
w_init = 20    # 突触的初始强度为0.5
taupre = 20     # 前突触的时间窗口
taupost = 20    # 后突触的时间窗口
apre = 0.1
apost = -0.12
# 仿真参数
sim_time = 500
dt = 1
# 权重矩阵：shape=(n_output, n_input)
w = np.ones((n_output, n_input)) * w_init  
# 膜电位：shape=(n_output,)
v = np.ones(n_output) * V_reset 
# 记录不应期，这里只需要给输出神经元不应期就行了，输入神经元不是真正的神经元，他就是个编码器，在图像识别中他负责亮的地方高电平，暗的地方低电平，本质是信号源
ref_counter = np.zeros(n_output, dtype = int)
# 痕迹
trace_pre = np.zeros(n_input)   #突触前，即突触与输入神经元的连接
trace_post = np.zeros(n_output) #突触后，即突触与输出神经元的连接
# 记录变量（用于可视化和分析）
input_spike_records = []   # 输入脉冲序列
output_spike_records = []  # 输出脉冲序列
v_records = []             # 输出层膜电位变化
w_records = [w.copy()]     # 权重变化（每步记录）
ref_records = []    # 不应期变化
# 设计模拟脉冲输入
def generate_input_spikes(t):
    spikes = np.zeros(n_input)
    # 输入0：0-100ms每10ms发放（对应输出0）
    if 0 <= t < 100 and t % 10 == 0:
        spikes[0] = 1
    # 输入2：100-200ms每10ms发放（对应输出1）
    if 100 <= t < 200 and t % 10 == 0:
        spikes[2] = 1
    # 输入3：随机发放（辅助输出1）
    if np.random.random() < 0.08:
        spikes[3] = 1
    # 输入4：0-100ms每20ms发放（辅助输出0）
    if 0 <= t < 100 and t % 20 == 0:
        spikes[4] = 1
    return spikes
# 进行STDP学习
for t in range (int (sim_time / dt)):
    s = generate_input_spikes (t)   # s记录了当前时间t的五个输入的脉冲
    input_spike_records.append(s)   # 记录
    # 前痕迹指数衰减，这里如果用最原始的方法一步一步累加来算LTP和LTD (下同) 的贡献就是On2，效率过于低下，有关的证明见AI吧
    trace_pre *= np.exp(-dt / taupre)
    trace_post *= np.exp(-dt / taupost)
    pre_spike_indices = np.where(s == 1) [0]
    if len(pre_spike_indices) > 0 :
        trace_pre[pre_spike_indices] += apre
        # 前脉冲引发LTD权重更新，核心，AI的第五步
        dw_LTD = np.outer(trace_post, s)
        w += dw_LTD
##   spi = []
##    for i in range (len(s)):
##        if s[i] == 1:
##           spi.append[i]
##            trace_pre[i] += apre
##            # 前脉冲引发LTD权重更新，核心，AI的第五步
##            dw_LTD = np.outer(trace_post, s)
##            w += dw_LTD
    # 更新后突出痕迹
#    trace_post *= np.exp(-dt / taupost)
    # 处理不应期
    # ref_counter = np.maximum(ref_counter - dt, 0)
#    no_ref_mask = []    # 非不应期的神经元掩码？这个是干什么的
    ref_counter = np.maximum(ref_counter - 1, 0)  # 直接用NumPy向量化递减，不用写for循环
    # 修复：no_ref_mask 是布尔数组
    no_ref_mask = (ref_counter == 0)
    I_input = np.dot(w, s)  # 输入电流，这里和1输入1输出有区别了，是多输入多输出
    # 更新膜电位，只有非不应期的才更新
    # LIF核心公式，写了好多遍的dV = (-(V[t - 1] - V_rest) + I_input) /tau_m * dt, V[t] = V[t - 1] + dV
    v[no_ref_mask] += (-(v[no_ref_mask] - V_rest) + I_input[no_ref_mask]) / tau_m * dt   # 这里的等号右边的v就是上一刻的等号左边的v
    # 检查是否发放脉冲
    output_spikes = np.zeros(n_output)
    spike_mask = (v >= V_th) & no_ref_mask
    post_spike_indices = np.where(spike_mask)[0]
    if (len (post_spike_indices) > 0):
        ref_counter[post_spike_indices] = t_ref
        output_spikes[post_spike_indices] = 1
        v[post_spike_indices] = V_reset
        # 这里就有后脉冲了，需要更前脉冲一样的操作
#        trace_post *= np.exp(-dt / taupost)
    output_spike_indices = np.where(output_spikes == 1)[0]
    if (len(output_spike_indices) > 0):
        trace_post[output_spike_indices] += apost
         # 后脉冲引发LTP权重更新，核心，AI的第五步
        dw_LTP = np.outer(output_spikes, trace_pre)
        w += dw_LTP
#   spi = []
#        for i in range (len(s)):
#            if s[i] == 1:
#               spi.append[i]
#                trace_post[i] += apost
#                # 后脉冲引发LTP权重更新，核心，AI的第五步
#                dw_LTP = np.outer(output_spikes, trace_pre)
#                w += dw_LTP
        # 更新前突出痕迹
#        trace_pre *= np.exp(-dt / taupre)
    # 记录状态
    output_spike_records.append(output_spikes.copy())
    v_records.append(v.copy())
    # 权重裁剪（防止溢出），把w限制在0到1之间
    w = np.clip(w, 0, 40)
    w_records.append(w.copy())
# vibe coding
# ====================== 可视化（验证STDP效果） ======================
input_spike_records = np.array(input_spike_records)
output_spike_records = np.array(output_spike_records)
v_records = np.array(v_records)
w_records = np.array(w_records)

plt.figure(figsize=(15, 12))
# 1. 输入脉冲光栅图
plt.subplot(4, 1, 1)
plt.title("Input Layer Spike Raster (5 neurons)")
for i in range(n_input):
    spike_times = np.where(input_spike_records[:, i] == 1)[0] * dt
    plt.scatter(spike_times, [i]*len(spike_times), color='blue', s=10)
plt.yticks(range(n_input), [f"Input {i}" for i in range(n_input)])
plt.ylabel("Neuron")
plt.xlim(0, sim_time)

# 2. 输出脉冲光栅图
plt.subplot(4, 1, 2)
plt.title("Output Layer Spike Raster (2 neurons)")
for i in range(n_output):
    spike_times = np.where(output_spike_records[:, i] == 1)[0] * dt
    plt.scatter(spike_times, [i]*len(spike_times), color='red', s=20)
plt.yticks(range(n_output), [f"Output {i}" for i in range(n_output)])
plt.ylabel("Neuron")
plt.xlim(0, sim_time)

# 3. 输出层膜电位
plt.subplot(4, 1, 3)
plt.title("Output Layer Membrane Potential (mV)")
plt.plot(np.arange(sim_time/dt)*dt, v_records[:, 0], label="Output 0", color='red')
plt.plot(np.arange(sim_time/dt)*dt, v_records[:, 1], label="Output 1", color='green')
plt.axhline(y=V_th, color='black', linestyle='--', label="Threshold (-60mV)")
plt.axhline(y=V_reset, color='gray', linestyle=':', label="Reset (-70mV)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.xlim(0, sim_time)

# 4. 权重变化
plt.subplot(4, 1, 4)
plt.title("Synapse Weight Changes (STDP Effect)")
plt.plot(np.arange(sim_time/dt+1)*dt, w_records[:, 0, 0], label="W[Output0][Input0] (LTP)", color='red')
plt.plot(np.arange(sim_time/dt+1)*dt, w_records[:, 1, 2], label="W[Output1][Input2] (LTP)", color='green')
plt.plot(np.arange(sim_time/dt+1)*dt, w_records[:, 0, 2], label="W[Output0][Input2] (LTD)", color='gray')
plt.axhline(y=w_init, color='black', linestyle='--', label="Initial Weight (0.5)")
plt.ylabel("Weight")
plt.xlabel("Time (ms)")
plt.legend()
plt.xlim(0, sim_time)

plt.tight_layout()
plt.show()

# 打印结果
print("="*60)
print(f"最终权重矩阵 W (2×5)：")
print(np.round(w, 2))
print("\n关键权重变化：")
print(f"Output0-Input0: {np.round(w[0,0], 2)} (预期上升，LTP)")
print(f"Output1-Input2: {np.round(w[1,2], 2)} (预期上升，LTP)")
print(f"Output0-Input2: {np.round(w[0,2], 2)} (预期下降，LTD)")
print("\n输出神经元发放次数：")
print(f"Output 0: {np.sum(output_spike_records[:, 0])} 次")
print(f"Output 1: {np.sum(output_spike_records[:, 1])} 次")