"""
================================================================================
 生产工程补齐 · 补5 · Prompt 版本管理（可追溯、可回滚、可 A/B、可灰度）
================================================================================
 补上「对 Prompt 做 Git 式版本管理」这一块。生产里 Prompt 就是"代码/配置"，改一句话可能让
 效果天差地别，必须版本化，不能散落在各处魔法字符串：
   ① 注册多版本：每个 prompt 存 name + version + 模板 + 备注，像 git commit 一样留痕。
   ② 渲染：模板用 {变量} 占位，render 时填参——把"提示结构"和"运行数据"分开。
   ③ 回滚/对比：出问题能一键切回上个版本；diff 看两版改了啥(可追溯)。
   ④ A/B & 灰度：线上按流量比例分发 v1/v2，配合补4的评测对比效果，再决定全量。
 为什么重要：Prompt 迭代频繁且影响大，没版本管理就没法"知道改了啥、坏了能回退、好了能复现"。
 纯 Python 实现(可选落盘 JSONL)，无外部依赖。
 跑：python3 补5_Prompt版本管理.py   （注：按用户要求本文件未在本机执行，仅作真实可跑代码）
================================================================================
"""
import difflib


class PromptRegistry:
    def __init__(self):
        self._store = {}      # name -> [ {version, template, note}, ... ] 按注册顺序
        self._current = {}    # name -> version(当前生效版本)

    def register(self, name, version, template, note=""):
        self._store.setdefault(name, []).append({"version": version, "template": template, "note": note})
        self._current[name] = version           # 新注册默认设为当前
        return self

    def _find(self, name, version):
        for item in self._store.get(name, []):
            if item["version"] == version:
                return item
        raise KeyError(f"{name}@{version} 不存在")

    def get(self, name, version=None):
        version = version or self._current[name]
        return self._find(name, version)["template"]

    def render(self, name, version=None, **kw):
        return self.get(name, version).format(**kw)     # {变量} 填参

    def rollback(self, name, version):
        self._find(name, version)                        # 校验存在
        self._current[name] = version                    # 切回历史版本
        return self

    def log(self, name):
        return [f"{i['version']}  {i['note']}" for i in self._store.get(name, [])]

    def diff(self, name, v1, v2):
        a = self._find(name, v1)["template"].splitlines()
        b = self._find(name, v2)["template"].splitlines()
        return "\n".join(difflib.unified_diff(a, b, v1, v2, lineterm=""))


def ab_pick(name, reg, user_id, ratio=0.5, versions=("v1", "v2")):
    """按 user_id 哈希稳定分流(同一用户总是同一版本)，做 A/B 灰度。"""
    bucket = (hash(user_id) % 100) / 100.0
    return versions[0] if bucket < ratio else versions[1]


def main():
    reg = PromptRegistry()
    reg.register("客服回复", "v1", "你是客服。根据资料回答：{context}\n问题：{q}", "初版")
    reg.register("客服回复", "v2", "你是专业客服，只依据资料、无据说不知道。\n资料：{context}\n问题：{q}\n请简洁回答。", "加防幻觉约束")

    r = reg.render("客服回复", context="7 天退货", q="能退吗?")     # 当前 v2
    assert "只依据资料" in r
    reg.rollback("客服回复", "v1")                                  # 回滚到 v1
    assert "只依据资料" not in reg.render("客服回复", context="x", q="y")
    d = reg.diff("客服回复", "v1", "v2")                            # 看两版差异
    picks = {ab_pick("客服回复", reg, f"user{i}") for i in range(20)}  # A/B 分流

    assert "v1" in reg.log("客服回复")[0] and d and picks
    print(f"✅ 补5 跑通：注册2版 + 渲染 + 回滚 + diff({len(d.splitlines())}行) + A/B分流命中{picks}")
    # 面试：Q Prompt 为什么要版本管理? A 改一句影响大,要可追溯/可回滚/可复现;
    #      Q 怎么灰度新 Prompt? A 按用户哈希稳定分流一部分流量到 v2,配合评测对比再全量。


if __name__ == "__main__":
    main()
