"""
================================================================================
 Deep RL Course · Chapter 7 · 多智能体系统(MARL)（学习笔记 · 非平稳演示 numpy 可跑）
================================================================================
 一句话：多个智能体同环境协作/竞争;每个 agent 的最优动作依赖别人 → 环境对单 agent 变"非平稳"。
 本章讲(纯 numpy 用一个协作博弈演示"收益取决于双方动作",无需库)：
   ① 收益矩阵:两 agent 同时选动作,回报由"组合"决定(不是各自独立)。
   ② 非平稳:对 agent A,环境(含 B 的策略)在变 → 普通单 agent 算法可能不收敛。
   ③ 分散式(各自训) vs 集中训练-分散执行(CTDE)。
 要点：多 agent 让环境非平稳,是 MARL 的核心难点;协作要"看组合收益"。
 跑：python3 chapter7_多智能体系统_学习笔记.py   （协作博弈纯 numpy 真跑)
================================================================================
"""
import numpy as np

# 协作收益矩阵:两 agent 都选动作1 时共赢(回报最高)
PAYOFF = np.array([[0, 0], [0, 2]])   # [a_A][a_B]


def joint_return(a_A, a_B):
    return PAYOFF[a_A, a_B]


def main():
    # 单看 A:选动作1 好不好,取决于 B 选什么(非平稳体现)
    a_given_B0 = [joint_return(a, 0) for a in (0, 1)]   # B=0 时 A 两动作回报
    a_given_B1 = [joint_return(a, 1) for a in (0, 1)]   # B=1 时
    assert max(a_given_B1) > max(a_given_B0)            # A 的最优依赖 B
    assert joint_return(1, 1) == 2                      # 双方协作(都选1)共赢
    print(f"✅ Ch7 跑通：A 的最优动作依赖 B(B=0→{a_given_B0}, B=1→{a_given_B1});双方选1 共赢=2")
    print("   → 对单 agent 环境非平稳(别人也在学),普通单 agent 算法难收敛;需 CTDE 等。")


if __name__ == "__main__":
    main()
