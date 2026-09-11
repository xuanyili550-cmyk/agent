import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

words = open('names.txt').read().splitlines()
chars=sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}
vocab_size = len(itos)
block_size = 3

def build_dataset(words):
    X,Y=[],[]
    for w in words:
        ctx=[0]*block_size
        for ch in w +'.':
            X.append(ctx)
            Y.append(stoi[ch])
            ctx=ctx[1:]+[stoi[ch]]
    return torch.tensor(X),torch.tensor(Y)

random.seed(42)
random.shuffle(words)
n1=int(0.8*len(words))
n2=int(0.9*len(words))
Xtr,Ytr=build_dataset(words[:n1])
Xdev,Ydev=build_dataset(words[n1:n2])


def init_params(n_embd=10,n_hidden=200, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((vocab_size, n_embd), generator=g)
    W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
    b1 = torch.randn(n_hidden, generator=g)
    W2 = torch.randn((n_hidden, vocab_size), generator=g)
    b2 = torch.randn(vocab_size, generator=g)
    params = [C, W1, b1, W2, b2]
    for p in params: p.requires_grad = True
    return params

def forward(params,Xb):
    C, W1, b1, W2, b2 = params
    emb=C[Xb]
    h=torch.tanh(emb.view(emb.shape[0],-1)@W1+b1)
    return h@W2+b2
params=init_params()
lre = torch.linspace(-3, 0, 1000)
lrs=10**lre
lr_log,loss_log=[],[]
for i in range(1000):
    ix=torch.randint(0,Xtr.shape[0],(32,))
    loss=F.cross_entropy(forward(params,Xtr[ix]),Ytr[ix])
    for p in params: p.grad= None
    loss.backward()
    lr=lrs[i].item()
    for p in params:p.data+=-lr*p.grad
    lr_log.append(lre[i].item())
    loss_log.append(loss.item())

plt.figure(figsize=(8,4))
plt.plot(lr_log,loss_log)
plt.xlabel('log10(learning rate)'); plt.ylabel('loss')
plt.title('LR range test'); plt.grid(True)
plt.show()

@torch.no_grad()
def full_loss(params,X,Y,batch=8192):
    tot=cnt=0
    for i in range(0,X.shape[0],batch):
        l=F.cross_entropy(forward(params,X[i:i+batch]),Y[i:i+batch],reduction='sum')
        tot+=l.item()
        cnt+=min(batch,X.shape[0]-i)
    return tot/cnt
params=init_params()
steps, tr_curve, va_curve = [], [], []
for i in range(8000):
    ix=torch.randint(0,Xtr.shape[0],(64,))
    loss=F.cross_entropy(forward(params,Xtr[ix]),Ytr[ix])
    for p in params: p.grad=None
    loss.backward()
    lr= 0.1 if i <5000 else 0.1
    for p in params:p.data+=-lr*p.grad
    if i%400==0:
        steps.append(i)
        tr_curve.append(full_loss(params,Xtr,Ytr))
        va_curve.append(full_loss(params,Xdev,Ydev))

plt.figure(figsize=(8, 4))
plt.plot(steps, tr_curve, label='train')
plt.plot(steps, va_curve, label='val')
plt.xlabel('step'); plt.ylabel('loss'); plt.legend()
plt.title('training / validation loss'); plt.grid(True)
plt.show()
print('最终 train loss=%.4f  val loss=%.4f' % (tr_curve[-1], va_curve[-1]))

params2 = init_params(n_embd=2)
for i in range(15000):
    ix=torch.randint(0,Xtr.shape[0],(64,))
    loss=F.cross_entropy(forward(params2,Xtr[ix]),Ytr[ix])
    for p in params2 : p.grad==None
    loss.backward()
    lr = 0.1 if i < 10000 else 0.01
    for p in params2: p.data += -lr * p.grad
C2=params2[0]
plt.figure(figsize=(8, 8))
plt.scatter(C2[:,0].data,C2[:, 1].data, s=200, color='steelblue')
for i in range(C2.shape[0]):
    plt.text(C2[i,0].item(), C2[i,1].item(), itos[i],
             ha='center', va='center', color='white')
plt.title('learned 2D character embeddings'); plt.grid(True)
plt.show()

@torch.no_grad()
def sample(params, n=300, temperature=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    name=[]
    for _ in range(n):
        ctx=[0]*block_size
        out=[]
        while True:
            logits=forward(params,torch.tensor([ctx]))/temperature
            p=F.softmax(logits,dim=1)
            ix=torch.multinomial(p,1,generator=g).item()
            if ix==0:break
            out.append(ix)
            ctx=ctx[1:]+[ix]
        name.append("".join(itos[i] for i in out))
    return name
temps = [0.5, 1.0, 1.5]
batches = {t: sample(params, 300, t, seed=7) for t in temps}
for t in temps:
    print(f'温度={t}:', batches[t][:5])

plt.figure(figsize=(8, 4))
for t in temps:
    lens = [len(nm) for nm in batches[t]]
    plt.hist(lens, bins=range(1, 15), alpha=0.5, label=f'T={t}')
plt.xlabel('name length'); plt.ylabel('count'); plt.legend()
plt.title('name length distribution by temperature'); plt.grid(True)
plt.show()