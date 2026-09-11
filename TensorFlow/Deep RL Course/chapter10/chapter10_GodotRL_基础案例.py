"""
 Deep RL Course · Ch10 · 基础案例：最小 env-agent 交互循环(纯 numpy,可跑)
 跑：python3 本文件
"""
pos, n, total = 0, 5, 0.0
for step in range(10):
    pos = min(n - 1, pos + 1)              # 策略:总向右
    r = 1.0 if pos == n - 1 else 0.0
    total += r
    if pos == n - 1:
        print(f"第 {step+1} 步到达目标,回报 {total}"); break
print("✅ RL 训练内核=reset→(动作→观测→奖励)循环;Godot/Gym 提供不同的 env")
