
# ============================================================================
# 案例 1：反向传播里必须用 += 累加梯度
#   每个 ValueOK 是一个标量自动微分节点：包住一个数字，并记住它是怎么算出来的，
#   好在反向传播时把梯度回传给它的输入。
# ============================================================================
class ValueOK:
    def __init__(self, data, _children=()):
        self.data = data                    # 这个节点的数值本身
        self._prev = set(_children)         # 产生我的输入节点（计算图里指向我的父节点），set 去重
        self._backward = lambda: None       # 把我的梯度分发给输入的函数；叶子节点无输入，默认空函数
        self.grad = 0.0                     # 损失对这个节点的偏导数，初始为 0

    def __add__(self, o):                   # 重载 +，让 a + b 能作用在 ValueOK 上
        o = o if isinstance(o, ValueOK) else ValueOK(o)   # 加数是普通数字就包成 ValueOK，保证能算梯度
        out = ValueOK(self.data + o.data, (self, o))      # 结果节点：数值相加，父节点记为 (self, o)

        def _b():
            self.grad += out.grad  # 累加      # 加法导数为 1，梯度原样回传；用 += 因为同一节点可能被用多次
            o.grad += out.grad

        out._backward = _b                  # 把反向函数挂到 out 上
        return out

    def backward(self):                     # 对最终输出节点调用，触发整张图的反向传播
        topo = []                           # 拓扑排序结果
        vis = set()                         # 记录已访问节点，避免重复

        def bt(v):                          # build topo：深度优先，先递归父节点，再把自己加进 topo
            if v not in vis:
                vis.add(v)
                for c in v._prev: bt(c)     # 保证父节点排在子节点前面
                topo.append(v)

        bt(self)                            # 从最终节点开始建拓扑序
        self.grad = 1.0                     # 最终输出对自己的导数为 1（dy/dy=1），反向传播的起点
        for n in reversed(topo): n._backward()  # 逆序（输出->输入）逐个调用 _backward，梯度层层回传

a = ValueOK(3.0)                            # 值为 3 的节点
(a + a).backward()                          # 算 a+a（=6）并反向；a 被用两次，+= 让梯度累加成 2
print("正确版  d(a+a)/da =", a.grad)          # 正确结果是 2（若用 = 而非 += 会错成 1）

import torch

at = torch.tensor(3.0, requires_grad=True)  # 用 PyTorch 做对照的“标准答案”，requires_grad 追踪梯度
(at + at).backward()
print("torch   d(a+a)/da =", at.grad.item())  # 同样是 2，验证自己写的引擎对了
# 👉 结论：所有 _backward 里都要用 += ；上面 cell 7 请把 = 改成 +=


### 案例 2：数值稳定的 softmax / 交叉熵（log-sum-exp 技巧）

#分类任务里 `exp()` 极易 **overflow**：`logits` 稍微大一点（比如 1000）`np.exp` 直接变 `inf → nan`。
#注意上面 cell 7 的 `tanh` 用了 `math.exp(2*x)`，`x` 大时同样会炸。

#**生产做法：** softmax 前先减去 `max(logits)`（结果不变，因为 softmax 平移不变），
#交叉熵用 `log-softmax = z - logsumexp(z)` 一步算，避免先 exp 再 log。这就是 PyTorch `cross_entropy` 内部做的事。

import numpy as np, torch

logits = np.array([1000., 1001., 1002.])        # 故意用超大 logits 触发溢出
naive = np.exp(logits)/np.exp(logits).sum()     # 朴素 softmax：exp(1000)=inf，inf/inf=nan
print("naive softmax :", naive)                 # 会打印出 nan

def softmax(z):
    z = z - z.max()                              # 关键：减最大值。softmax 平移不变，结果不变但不再溢出
    e = np.exp(z); return e/e.sum()
def log_softmax(z):
    z = z - z.max()
    return z - np.log(np.exp(z).sum())           # log_softmax = z - logsumexp(z)，避免先 exp 再 log
def cross_entropy(logits, target):               # target 是类别下标
    return -log_softmax(logits)[target]          # 交叉熵 = 负的“正确类别的 log 概率”

print("stable softmax:", softmax(logits))        # 输出正常概率，约 [0.09, 0.24, 0.66]
print("cross_entropy :", cross_entropy(logits, target=2))
tl = torch.tensor([[1000.,1001.,1002.]])
print("torch CE      :", torch.nn.functional.cross_entropy(tl, torch.tensor([2])).item())  # PyTorch 对照，内部同样做稳定处理


## 案例 3：PyTorch 生产版训练循环（对应上面 cell 21–25 的手写 MLP）

#上面手写训练循环缺了生产里必备的几件事，这里补全，用的是同一个 4 样本玩具数据：

#- `optimizer.zero_grad()`：**每步清零梯度**（否则会累加——正是案例 1 的 `+=` 语义）
#- `Adam` 优化器 + **学习率调度器**（`StepLR`）
#- **梯度裁剪** `clip_grad_norm_`：防梯度爆炸（RNN/Transformer 必备）
#- `model.train()` / `model.eval()`：切换 dropout / batchnorm 行为

import torch
torch.manual_seed(0)                        # 固定随机种子，结果可复现

X = torch.tensor([[2.0,3.0,-1.0],[3.0,-1.0,0.5],[0.5,1.0,1.0],[1.0,1.0,-1.0]])  # 4 个样本，每个 3 维特征
Y = torch.tensor([1.0,-1.0,-1.0,1.0]).unsqueeze(1)   # 4 个标签；unsqueeze(1) 把 (4,) 变 (4,1) 对齐模型输出

model = torch.nn.Sequential(
    torch.nn.Linear(3,4), torch.nn.Tanh(),  # 3->4 线性层 + tanh 激活
    torch.nn.Linear(4,4), torch.nn.Tanh(),  # 4->4
    torch.nn.Linear(4,1), torch.nn.Tanh(),  # 4->1 输出
)                                       # 等价于 MLP(3,[4,4,1])
opt   = torch.optim.Adam(model.parameters(), lr=0.05)                      # Adam 优化器，比朴素梯度下降收敛更快更稳
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5)      # 每 40 步学习率乘 0.5（衰减），后期更精细
lossf = torch.nn.MSELoss()                                                 # 均方误差损失

for epoch in range(80):                 # 训练 80 轮
    model.train()                       # 切训练模式（影响 dropout/batchnorm；本例没有但是好习惯）
    opt.zero_grad()                     # 1) 清零（对应手写的 p.grad=0.0）；PyTorch 梯度默认累加，每步必须先清零
    loss = lossf(model(X), Y)           # 2) 前向 + 计算损失
    loss.backward()                     # 3) 反向，算出每个参数的梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 4) 梯度裁剪：范数限制到 <=1.0，防梯度爆炸
    opt.step()                          # 5) 更新（对应 p.data += -lr*p.grad）
    sched.step()                        # 6) 调整学习率，到点就衰减
    if epoch % 20 == 0:                 # 每 20 轮打印一次损失和当前学习率，观察收敛
        print(f"epoch {epoch:3d}  loss {loss.item():.5f}  lr {sched.get_last_lr()[0]:.4f}")

model.eval()                            # 切评估模式
with torch.no_grad():                   # 推理不需要梯度，关闭追踪省内存更快
    print("preds:", [round(v,3) for v in model(X).squeeze().tolist()])   # 最终预测，应接近目标 [1,-1,-1,1]


## 案例 4：梯度检验（gradient checking）—— 上线自定义算子前的标准验证

# 自己写反向传播（案例 1 那种 `_backward`）怎么确认没写错？生产标准做法是**数值梯度对照**：
# 用有限差分 `(f(x+ε) − f(x−ε)) / 2ε` 近似真实导数，和你解析求的梯度比 **相对误差**，`< 1e-5` 才敢信。
# 这正是本 notebook 最前面 cell 4–6 那个 `h=0.0001` 求斜率的思路，只不过用它来**验证 autograd**。

import math

class V:  # 带 tanh 的正确版 Value（+= 累加）
    def __init__(self, data, _children=()):
        self.data=data; self._prev=set(_children); self._backward=lambda:None; self.grad=0.0   # 同案例 1 的节点结构
    def __add__(self, o):
        o=o if isinstance(o,V) else V(o); out=V(self.data+o.data,(self,o))
        def _b(): self.grad+=out.grad; o.grad+=out.grad   # 加法：梯度原样传给两个输入，用 +=
        out._backward=_b; return out
    def __mul__(self, o):
        o=o if isinstance(o,V) else V(o); out=V(self.data*o.data,(self,o))
        def _b(): self.grad+=o.data*out.grad; o.grad+=self.data*out.grad   # 乘法链式法则：对 self 导数是 o.data，对 o 导数是 self.data
        out._backward=_b; return out
    def tanh(self):
        t=math.tanh(self.data); out=V(t,(self,))
        def _b(): self.grad+=(1-t*t)*out.grad             # tanh 导数是 1-tanh^2，即 (1-t*t)
        out._backward=_b; return out
    def backward(self):                                   # 拓扑排序 + 逆序反向，逻辑同案例 1
        topo=[]; vis=set()
        def bt(v):
            if v not in vis:
                vis.add(v)
                for c in v._prev: bt(c)
                topo.append(v)
        bt(self); self.grad=1.0
        for n in reversed(topo): n._backward()

def forward(xval):                 # 任意标量函数 f(x) = tanh(2x + 1)
    x=V(xval); return (x*2 + 1).tanh(), x   # 同时返回结果和输入节点 x（好取它的梯度）

x0=0.7                             # 测试点 x=0.7
y, x = forward(x0); y.backward()   # 前向 + 反向传播
analytic = x.grad                  # 自己的引擎算出的解析梯度
eps=1e-6                           # 微小扰动 ε
numeric = (math.tanh(2*(x0+eps)+1) - math.tanh(2*(x0-eps)+1))/(2*eps)   # 中心差分数值梯度，不依赖自己的引擎
rel_err = abs(analytic-numeric)/max(1e-12, abs(numeric))               # 相对误差；分母加 max(1e-12,..) 防除 0
print(f"analytic={analytic:.8f}  numeric={numeric:.8f}  rel_err={rel_err:.2e}")
assert rel_err < 1e-5, "梯度写错了！"    # 相对误差 <1e-5 才算通过，否则报错
print("✅ 梯度检验通过")



## 案例 5：向量化 —— 从「标量 Value 引擎」到「张量」（生产为什么不用逐元素求导）

#micrograd 每个数都是一个 Python 对象、每个 `+`/`*` 都建一个节点：教学清晰，但 **N×D×H 规模会创建上百万个对象**，慢几个数量级。
#生产里一层线性层就是一个矩阵乘 `Y = X @ W`，前向/反向都用矩阵运算一次算完（下面手写反向，和链式法则完全一致）：

#- forward:  `Y = X @ W`
#- backward: `dW = Xᵀ @ dY`，`dX = dY @ Wᵀ`

import numpy as np, time

N, D, H = 256, 64, 128             # 批大小 256、输入维度 64、输出维度 128
rng = np.random.default_rng(0)     # 固定种子的随机数生成器
X = rng.standard_normal((N, D))    # 随机输入 256x64
W = rng.standard_normal((D, H))    # 随机权重 64x128

t0 = time.time()                   # 计时开始
#@ 是 Python 3.5+ 引入的运算符（PEP 465），专门用来表示矩阵乘法。NumPy、PyTorch 等库都支持它：
#X @ W        # 等价于 np.matmul(X, W)  或  X.matmul(W) (PyTorch)

Y  = X @ W                      # 前向：一次矩阵乘 = N*D*H 次标量乘
dY = np.ones_like(Y)            # 假设上游梯度全为 1（相当于损失是 Y.sum()）
dW = X.T @ dY                   # 反向（和 Value 的 __mul__ 链式法则一致，只是批量化）
dX = dY @ W.T
dt = (time.time()-t0)*1000         # 耗时（毫秒）
print(f"向量化 前向+反向: {dt:.2f} ms   dW{dW.shape}  dX{dX.shape}")
print(f"若用标量 Value 引擎需创建 ~{N*D*H:,} 个节点对象，慢几个数量级、内存也扛不住")   # 约 210 万个对象

# 数值梯度抽查一个 W[i,j]，确认手写 dW 正确
i, j = 3, 5
eps = 1e-5
Wp = W.copy(); Wp[i,j]+=eps; Wm = W.copy(); Wm[i,j]-=eps   # 把 W[3,5] 分别 +ε 和 -ε
num = ((X@Wp).sum() - (X@Wm).sum())/(2*eps)   # loss=Y.sum() 时 dY=全1；中心差分得数值梯度
print(f"dW[{i},{j}] analytic={dW[i,j]:.4f}  numeric={num:.4f}")   # 与手写 dW[3,5] 对比，一致即正确
