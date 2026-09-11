"""
 Deep RL Course · Ch1 · 基础案例：折扣累计回报(纯 numpy,可跑)
 跑：python3 本文件
"""
def discounted_return(rewards, gamma=0.9):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G; out.append(G)
    return out[::-1]

print("奖励序列 [1,0,0,1] 的逐步回报(γ=0.9):", [round(x,3) for x in discounted_return([1,0,0,1])])
print("✅ G_t = r_t + γ·G_{t+1},RL 要最大化的就是它")
