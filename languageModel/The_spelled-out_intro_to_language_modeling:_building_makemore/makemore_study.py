import math

import  torch
import torch.nn as nn
import torch.nn.functional as F
import random

device='cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu')
words = open('names.txt', 'r').read().splitlines()
chars=sorted(list(set(''.join(words))))

stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itso={i:s for s ,i in stoi.items()}
vocab_size=len(itso)
block_size=3
print('词表大小:', vocab_size, '  样本名字数:', len(words))  # 打印基本信息

# --- 把名字列表转成 (X, Y) 张量数据 ---
def build_dataset(words):                            # 输入名字列表,输出(输入上下文, 目标字符)
    X, Y = [], []                                    # X=上下文, Y=目标
    for w in words:                                  # 遍历名字
        context = [0] * block_size                   # 初始上下文全是 '.'
        for ch in w + '.':                          # 遍历字符(末尾补 '.')
            X.append(context)                        # 记录上下文
            Y.append(stoi[ch])                       # 记录目标字符
            context = context[1:] + [stoi[ch]]      # 滑动窗口更新上下文
    return torch.tensor(X), torch.tensor(Y)          # 转成张量返回

random.seed(42)
random.shuffle(words)
n1=int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev =build_dataset(words[n1:])
Xtr, Ytr = Xtr.to(device),Ytr.to(device)
Xdev, Ydev = Xdev.to(device), Ydev.to(device)
print('训练样本:', Xtr.shape[0], ' 验证样本:', Xdev.shape[0])


## 案例 1：把模型封装成 `nn.Module` 类

class MakeMoreMLP(nn.Module):
    def __init__(self,vocab_size,block_size,n_embd=10, n_hidden=200):
        super().__init__()
        self.block_size = block_size
        self.emb=nn.Embedding(vocab_size,n_embd)
        self.fc1=nn.Linear(block_size*n_embd,n_hidden)
        self.fc2=nn.Linear(n_hidden,vocab_size)

    def forward(self,x):
        emb=self.emb(x)
        emb=emb.view(emb.shape[0],-1)
        h=torch.tanh(self.fc1(emb))
        logits=self.fc2(h)
        return logits
torch.manual_seed(42)
model=MakeMoreMLP(vocab_size,block_size).to(device)
print(model)
print('参数总量:', sum(p.numel() for p in model.parameters()))

#带验证监控 + 早停 + 保存最优的训练循环

@torch.no_grad()
def eval_loss(model,X,Y,batch=4096):
    model.eval()
    total,cnt=0.0,0
    for i in range(0,X.shape[0],batch):
        xb, yb =X[i:i+batch],Y[i:i+batch]
        loss=F.cross_entropy(model(xb),yb,reduction='sum')
        total+=loss.item()
        cnt+=xb.shape[0]
    model.train()
    return total/cnt

optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

max_steps   = 5000        # 最多训练步数
batch_size  = 64          # 每步 mini-batch 大小
eval_every  = 500         # 每隔多少步评估一次验证集
patience    = 3           # 早停:验证 loss 连续这么多次没改善就停
best_val    = float('inf')# 记录最优验证 loss
bad_count   = 0           # 连续没改善的次数
best_state  = None        # 保存最优时刻的权重快照


for step in range(1,max_steps+1):
    ix=torch.randint(0,Xtr.shape[0],(batch_size,),device=device)
    xb,yb=Xtr[ix],Ytr[ix]
    logits=model(xb)
    loss=F.cross_entropy(logits,yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step%eval_every==0:
        val=eval_loss(model,Xdev,Ydev)
        print(f'step {step:5d} | train {loss.item():.4f} | val {val:.4f}')
        if val < best_val -1e-4:
            best_val=val
            best_state={k : v.detach().cpu().clone() for k ,v in model.state_dict().items()}
            bad_count=0
        else:
            bad_count+=1
            if bad_count>=patience:
                print(f'验证 loss 连续 {patience} 次未改善,提前停止')
                break

if best_state is not  None:
    model.load_state_dict(best_state)
print('最优验证 loss:', round(best_val, 4))

#保存 / 加载模型用于上线推理
CKPT = 'makemore_model.pt'
torch.save({
    'state_dict': model.state_dict(),
    'stoi': stoi, 'itso': itso,
    'vocab_size': vocab_size, 'block_size': block_size,
}, CKPT)
ckpt = torch.load(CKPT, map_location=device)
print(ckpt.keys())
infer_model = MakeMoreMLP(ckpt['vocab_size'], ckpt['block_size']).to(device)
infer_model.load_state_dict(ckpt['state_dict'])
infer_model.eval()
itos_loaded = ckpt['itso']
print('加载完成,可用于推理')

#可控采样——温度 + top-k + 指定前缀续写
@torch.no_grad()
def generate(model, itso, n=10, temperature=1.0, top_k=None, prefix='', seed=None):
    model.eval()
    stoi_local={s:i for i,s in itso.items() }
    g=torch.Generator(device=device)
    if seed is not None:g.manual_seed(seed)
    results = []
    for _ in range(n):
        context=[0]*model.block_size
        out = []
        for ch in prefix:
            ix=stoi_local[ch]
            out.append(ix)
            context=context[1:]+[ix]
        while True:
            x = torch.tensor([context], device=device)
            logits=model(x)
            logits=logits / temperature
            if top_k is not None:
                v,_=torch.topk(logits,top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs=F.softmax(logits,dim=1)
            ix=torch.multinomial(probs,1,generator=g).item()
            if ix==0:
                break
            out.append(ix)
            context = context[1:] + [ix]
        results.append(''.join(itso[i] for i in out))
    return results
print("温度=1.0 (默认):     ",generate(infer_model,itos_loaded,5,seed=1))
print('温度=0.5 (更稳重):   ', generate(infer_model, itos_loaded, 5, temperature=0.5, seed=1))
print('温度=1.3 (更狂野):   ', generate(infer_model, itos_loaded, 5, temperature=1.3, seed=1))
print('top_k=5 (只选高概率):', generate(infer_model, itos_loaded, 5, top_k=5, seed=1))
print("前缀 'ka' 续写:      ", generate(infer_model, itos_loaded, 5, prefix='ka', seed=1))

#批量生成“全新”结果 + 评估指标
existing = set(words)
raw=generate(infer_model, itos_loaded, n=200, top_k=10, seed=7)
fresh=[]
seen=set()
for name in raw:
    if len(name) < 2: continue
    if name in existing: continue
    if name in seen: continue
    seen.add(name)
    fresh.append(name)
print(fresh[:15])
val_loss=eval_loss(infer_model,Xdev,Ydev)
perplexity=math.exp(val_loss)
print(f'验证集 loss={val_loss:.4f}  perplexity={perplexity:.2f}')



