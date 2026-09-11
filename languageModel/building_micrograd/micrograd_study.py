import numpy as np, torch, time
import math

class Value:
    def __init__(self,data,_children=()):
        self.data=data
        self._prev=set(_children)
        self._backward=lambda :None
        self.grad=0.0

    def __add__(self, o):
        o=o if isinstance(o,Value) else Value(o)
        out =Value(self.data+o.data,(self,o))
        def _b():
            self.grad+=out.grad
            o.grad+=out.grad
        out._backward=_b
        return out

    def backward(self):
        topo=[]
        vis=set()
        def bt(v):
            if v not in vis:
                vis.add(v)
                for c in v._prev:bt(c)
                topo.append(v)

        bt(self)
        self.grad = 1.0
        for n in reversed(topo): n._backward()
a=Value(3.0)
(a + a).backward()
print(a.grad)

at=torch.tensor(3.0,requires_grad=True)
(at+at).backward()
print("torch   d(a+a)/da =", at.grad.item())


logits = np.array([1000., 1001., 1002.])
naive = np.exp(logits)/np.exp(logits).sum()

def softmax(z):
    z=z-z.max()
    e=np.exp(z)
    return e/e.sum()

def log_softmax(z):
    z = z - z.max()
    return z-np.log(np.exp(z).sum())

def cross_entropy(logits, target):               # target 是类别下标
    return -log_softmax(logits)[target]

print("stable softmax:", softmax(logits))
print("cross_entropy :", cross_entropy(logits, target=2))
tl = torch.tensor([[1000.,1001.,1002.]])
print("torch CE      :", torch.nn.functional.cross_entropy(tl, torch.tensor([2])).item())


torch.manual_seed(0)
X = torch.tensor([[2.0,3.0,-1.0],[3.0,-1.0,0.5],[0.5,1.0,1.0],[1.0,1.0,-1.0]])
Y = torch.tensor([1.0,-1.0,-1.0,1.0]).unsqueeze(1)
model = torch.nn.Sequential(
    torch.nn.Linear(3,4), torch.nn.Tanh(),
    torch.nn.Linear(4,4), torch.nn.Tanh(),
    torch.nn.Linear(4,1), torch.nn.Tanh(),
)                                       # 等价于 MLP(3,[4,4,1])
opt   = torch.optim.Adam(model.parameters(), lr=0.05)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5)
lossf = torch.nn.MSELoss()

for epoch in range(80):
    model.train()
    opt.zero_grad()                     # 1) 清零（对应手写的 p.grad=0.0）
    loss = lossf(model(X), Y)           # 2) 前向 + 计算损失
    loss.backward()                     # 3) 反向
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 4) 梯度裁剪
    opt.step()                          # 5) 更新（对应 p.data += -lr*p.grad）
    sched.step()                        # 6) 调整学习率
    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.5f}  lr {sched.get_last_lr()[0]:.4f}")

model.eval()
with torch.no_grad():
    print("preds:", [round(v,3) for v in model(X).squeeze().tolist()])


class V:  # 带 tanh 的正确版 Value（+= 累加）
    def __init__(self, data, _children=()):
        self.data=data; self._prev=set(_children); self._backward=lambda:None; self.grad=0.0
    def __add__(self, o):
        o=o if isinstance(o,V) else V(o); out=V(self.data+o.data,(self,o))
        def _b(): self.grad+=out.grad; o.grad+=out.grad
        out._backward=_b; return out
    def __mul__(self, o):
        o=o if isinstance(o,V) else V(o); out=V(self.data*o.data,(self,o))
        def _b(): self.grad+=o.data*out.grad; o.grad+=self.data*out.grad
        out._backward=_b; return out
    def tanh(self):
        t=math.tanh(self.data); out=V(t,(self,))
        def _b(): self.grad+=(1-t*t)*out.grad
        out._backward=_b; return out
    def backward(self):
        topo=[]; vis=set()
        def bt(v):
            if v not in vis:
                vis.add(v)
                for c in v._prev: bt(c)
                topo.append(v)
        bt(self); self.grad=1.0
        for n in reversed(topo): n._backward()

def forward(xval):                 # 任意标量函数 f(x) = tanh(2x + 1)
    x=V(xval); return (x*2 + 1).tanh(), x

x0=0.7
y, x = forward(x0); y.backward()
analytic = x.grad
eps=1e-6
numeric = (math.tanh(2*(x0+eps)+1) - math.tanh(2*(x0-eps)+1))/(2*eps)
rel_err = abs(analytic-numeric)/max(1e-12, abs(numeric))
print(f"analytic={analytic:.8f}  numeric={numeric:.8f}  rel_err={rel_err:.2e}")
assert rel_err < 1e-5, "梯度写错了！"
print("✅ 梯度检验通过")


N, D, H = 256, 64, 128
rng = np.random.default_rng(0)
X = rng.standard_normal((N, D))
W = rng.standard_normal((D, H))

t0 = time.time()
Y  = X @ W                      # 前向：一次矩阵乘 = N*D*H 次标量乘
dY = np.ones_like(Y)            # 假设上游梯度
dW = X.T @ dY                   # 反向（和 Value 的 __mul__ 链式法则一致，只是批量化）
dX = dY @ W.T
dt = (time.time()-t0)*1000
print(f"向量化 前向+反向: {dt:.2f} ms   dW{dW.shape}  dX{dX.shape}")
print(f"若用标量 Value 引擎需创建 ~{N*D*H:,} 个节点对象，慢几个数量级、内存也扛不住")

# 数值梯度抽查一个 W[i,j]，确认手写 dW 正确
i, j = 3, 5
eps = 1e-5
Wp = W.copy(); Wp[i,j]+=eps; Wm = W.copy(); Wm[i,j]-=eps
num = ((X@Wp).sum() - (X@Wm).sum())/(2*eps)   # loss=Y.sum() 时 dY=全1
print(f"dW[{i},{j}] analytic={dW[i,j]:.4f}  numeric={num:.4f}")
