import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import spikegen
import matplotlib.pyplot as plt

# 超参数设置
batch_size = 32     # 训练批次设置，训练的规模，小规模的数据训练更快
T_rate_coding = 10  # rate coding的时间窗口
n_input = 784       # 对应MNIST的28*28像素
n_hidden= 100       # 隐藏层神经元
n_output = 10       # 输出层神经元
learn_rate_stdp = 0.001  # stdp学习率，用此来调节突触强度，这个参数是用来调节input和hidden之间的强度的
learn_rate_supervisioned = 0.01     #监督学习学习率，同上，这个用于调节hidden和output之间的强度，最好比stdp权重大一点
num_epocs = 3       # 训练轮数，这里训练三轮

# 设备设置：有GPU用GPU，没有用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# vibe coding
# ====================== 2. 加载MNIST数据集（只取小部分，训练快） ======================
transform = transforms.Compose([
    transforms.ToTensor(),  # 转成Tensor，这里我不是很理解
    transforms.Normalize((0,), (1,))  # 归一化到[0,1]，正好对应Rate Coding的脉冲概率
])

# 加载训练集和测试集，只取前1000张训练、200张测试，验证用足够
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_dataset = torch.utils.data.Subset(test_dataset, range(200))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 最简单的rate coding函数
def rate_coding (img, t_steps):     # 输入为img(batch_size, 1, 28, 28), t_steps是时间步，所有脉冲在t_steps里发送
    img_flat = img.view(img.size(0), -1).to(device)     # pytorch自带的函数，把张量的图像平展成7百多个像素
    sent_spike = spikegen.rate(img_flat, num_steps = t_steps, gain = 1) # snntorch自带的函数，把imgflat的各个像素按照灰度值生成对应的脉冲，灰度值越大，脉冲概率就越大，在t_steps里面的脉冲数就越多
    return sent_spike
# 定义带STDP的SNN模型
class STDP_SNN(nn.Module) :
    def __init__(self, n_input, n_hidden, n_output, beta = 0.95):
        super().__init__()   # 继承父类模板
        self.fc1 = nn.Linear(n_input, n_hidden, bias = False)   # 输入层和隐藏层之间的的全连接层，STDP更新这一层，直接给输入层784个神经元和隐藏层100个神经元全部连接起来，到时候直接用矩阵乘法就行了，更方便
        self.fc2 = nn.Linear(n_hidden, n_output, bias = False)  # 同上，监督学习更新这一层
        self.lif1 = snn.Leaky(beta = beta)      # 每一层fc连接的Lif神经元
        self.lif2 = snn.Leaky(beta = beta)
        # 初始化权重：小一点，避免初始脉冲太多
        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.01)

    def forward (self, spike_seq) : #spike_seq是输入脉冲序列
        mem1 = self.lif1.init_leaky()   # 初始化膜电位，等价于我在LIF_5x2中的v
        mem2 = self.lif2.init_leaky()
        spike_rec1 = []  # 用于记录隐藏层输出的脉冲
        spike_rec2 = []  # 记录输出层的脉冲
        for step in range(T_rate_coding):  # 时间步循环（从 0 到 T-1）
            # 输入层到隐藏层
            cur1 = self.fc1(spike_seq[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            # 隐藏层到输出层
            cur2 = self.fc2(spike_seq[spk1])
            spk2, mem2 = self.lif2(cur2, mem2)
            spike_rec1.append(spk1)
            spike_rec2.append(spk2)
        # 转成Tensor：(time_steps, batch_size, neurons)
        spk1_rec = torch.stack(spk1_rec)
        spk2_rec = torch.stack(spk2_rec)
        # 记录每个输出层神经元在这个t_rate_code的总时间里
        output_spike_count = spk2_rec.sum(dim = 0)
        return output_spike_count
# STDP学习
def stdp (fc1_w, input_spike, hidden_spike, lr = 0.001):    # fc1权重，输入层脉冲顺序，隐藏层脉冲顺序，stdp学习率
    # 全部转化为浮点数方便计算
    input_spike = input_spike.float()
    hidden_spike = hidden_spike.float()
    # 更新突触痕迹
    '''
    回忆我在5*2的里面是如何写trace的：
    首先trace是用于处理LTP和LTD对权重的贡献的，由于直接暴搜以前所发送的所有脉冲并对其累加的时间复杂度过大，脉冲一大就容易崩溃，所以有了天才的trace用来替代
    有
    trace_pre *= np.exp(-dt / taupre)
    trace_post *= np.exp(-dt / taupost)
    if len(pre_spike_indices) > 0 :
        trace_pre[pre_spike_indices] += apre
        # 前脉冲引发LTD权重更新，核心，AI的第五步
        dw_LTD = np.outer(trace_post, s)
        w += dw_LTD
    。。。
    。。。
    。。。
    if (len(output_spike_indices) > 0):
        trace_post[output_spike_indices] += apost
         # 后脉冲引发LTP权重更新，核心，AI的第五步
        dw_LTP = np.outer(output_spikes, trace_pre)
        w += dw_LTP
    但是用AI给出的写法好像极其优化，简明易懂
    '''
    Alpha = 0.9   # 对应到之前的写法, alpha = np/exp(-dt / taupre)
    apre = apost = 1
    trace_pre = torch.zeros_like(input_spike)   # 初始化变量
    trace_post = torch.zeros_like(hidden_spike)
    for i in range(1, T_rate_coding):
        trace_pre[i] = trace_pre[i - 1] * Alpha + input_spike[i] * apre
    for i in range(1, T_rate_coding):
        trace_post[i] = trace_post[i - 1] * Alpha + hidden_spike[i] * apost 
    # 计算权重更新，这里Ai告诉了我torch里面叫爱因斯坦求和的一个优秀的写法
    # 计算权重更新：
    # 1. 突触前→突触后：pre_trace * spk_post → 权重增加
    # 2. 突触后→突触前：post_trace * spk_pre → 权重减少
    delta_w_pos = torch.einsum("tbi,tbh->hbi", trace_pre, hidden_spike).mean(dim=1)  # 对批次平均
    delta_w_neg = torch.einsum("tbh,tbi->hbi", trace_post, input_spike).mean(dim=1)
    fc1_w += delta_w_pos
    fc1_w -= delta_w_neg
    fc1_w *= lr
    if (fc1_w >= 1) :
        fc1_w = 1
    elif (fc1_w <= -1):
        fc1_w = -1
    return fc1_w
print(1111)
# vibe coding
# ====================== 初始化模型和优化器 ======================
model = STDP_SNN(n_input, n_hidden, n_output).to(device)
# 只对隐藏层→输出层用监督优化器（STDP手动更新输入层→隐藏层）
optimizer = optim.Adam(model.fc2.parameters(), lr=learn_rate_supervisioned)
criterion = nn.CrossEntropyLoss()  # 分类用交叉熵损失
# vibe coding over
# 训练流程
print("开始训练")   # 提示开始训练
train_losses = []   # 用于记录损失
test_accs = []      # 用于记录每一轮训练精确度
for i in range(num_epocs):
    model.train()   # 其实没啥用，但以后升级的时候可能有用
    total_loss = 0  # 记录总损失
    for batch_idx, (date, target) in enumerate(train_loader):
        date, target = date.to(device), target.to(device)   # 把date（图像数据）和target（实际数字）的信息都放到用于训练的device（cpu/gpu）上
        spike_seq = rate_coding(date, T_rate_coding)        # 将图像编码化成频率编码
        # 前向传播，得到隐藏层和输出层的脉冲序列
        spk1_rec, output_spk = model(spike_seq)
        # 手动更新 fc1：用模块 5 的 STDP 规则，传入 fc1 当前权重、输入层脉冲、隐藏层脉冲，得到更新后的权重，赋值给 fc1
        model.fc1.weight.data = stdp(
            model.fc1.weight.data,
            spike_seq,
            spk1_rec,
            lr = learn_rate_stdp
        )
        # 监督更新 fc2，上面优化器已经绑定fc2，所以优化器会自己来更新fc2，我只需提供参数即可
        optimizer.zero_grad()   # 优化器的梯度清零
        loss = criterion(output_spk, target)    # 计算交叉熵损失？这里没太懂什么意思，但大致应该是计算损失
        loss.backward()     # 反向传播，把那些梯度传播给前面的神经元（突触），反向传播是不是是把这次离标准答案的差距（损失）传给前面的那些步骤，然后修改前面的权重（以前的突触强度），并且提供差距有多大，下一步要怎么改
        optimizer.step()    # 反向传播就是提供梯度，然后step就是依据提供的梯度来修改突触强度

        total_loss += loss.item()   # 计算总损失
    avg_loss= total_loss / len(train_loader)
    train_loader.append(avg_loss)
    # 遍历每个批次测试
    model.eval()
    correct = 0     # 正确数
    total = 0       # 总数
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            spike_seq = rate_coding(data, T_rate_coding)
            _, soutput = model(spike_seq)   # 只需向前传播就行，只用看输入，不需要训练了
            _, pre_ans = torch.max(soutput.data, 1)
            total += target.size(0)
            correct += (pre_ans == target).sum().item()
    
    test_acc = 100 * correct / total
    test_accs.append(test_acc)
    
    # 打印结果
    print(f"Epoch [{i+1}/{num_epocs}], Train Loss: {avg_loss:.4f}, Test Acc: {test_acc:.2f}%")

# ====================== 9. 可视化结果（看训练过程） ======================
plt.figure(figsize=(12, 4))

# 训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# 测试准确率曲线
plt.subplot(1, 2, 2)
plt.plot(test_accs, label='Test Acc', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# ====================== 10. 可视化STDP更新后的权重（看效果） ======================
# 取输入层→隐藏层的前10个神经元的权重，可视化成28×28的图像
weights = model.fc1.weight.data[:10].cpu().numpy()
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(weights[i].reshape(28, 28), cmap='gray')
    plt.title(f'Hidden Neuron {i+1}')
    plt.axis('off')
plt.tight_layout()
plt.show()

print("训练完成！")