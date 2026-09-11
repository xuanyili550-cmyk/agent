"""
================================================================================
 MCP Course · Chapter 4 · 案例：连“多能力服务器”的 MCP Agent（工具 + 资源 + 提示 一起用）
================================================================================
 chapter4 的笔记里，Agent 只连了 chapter2 那个“纯工具”服务器。本案例更完整：
 我们在同一个文件里内置一个【同时暴露 工具 / 资源 / 提示 三种原语】的服务器，
 再写一个 Agent，演示真实产品里 AI 用 MCP 的完整闭环——不只是“调一个工具”，而是：

   ① 连接 & 能力发现：initialize 后，一次性发现 tools/resources/prompts（AI 先摸清“有啥能力”）。
   ② 多步规划：针对一个问题，Agent 可能要【先读资源拿上下文】→【再调工具算数】→【最后套提示模板产出】。
   ③ 用资源做 RAG 式检索：search 工具在“知识库资源”里检索，把命中的资源正文作为回答依据。
   ④ 用提示统一产出格式：最终答案套用服务器提供的 answer 提示模板。

 这就是 Claude Desktop / Cursor 这类 Host 背后的机制：LLM 看着 MCP 暴露的能力清单，
 自己决定“这一步该读资源、调工具，还是套提示”。本案例用【规则】替代 LLM 做决策，
 逻辑结构与真实 Agent 一致（真实里是把能力清单 + 问题发给 LLM，让它输出下一步动作）。

 运行：
   python3 chapter4_多能力MCP_Agent案例.py           # 跑 Agent（自动 spawn 内置服务器）
   python3 chapter4_多能力MCP_Agent案例.py --server   # 仅作为“多能力 MCP 服务器”运行
================================================================================
"""

import json
import re
import subprocess
import sys

# ============================== 服务器侧：三种原语一次备齐 ==============================
# 知识库：既作为“资源”暴露，也被 search 工具检索（体现资源=上下文来源）
KB = {
    "refund": "退款政策：7 天内无理由退款，需保留原始包装；超过 7 天按折旧计算。",
    "shipping": "配送政策：下单 48 小时内发货，偏远地区 +2 天；满 99 元包邮。",
    "warranty": "保修政策：整机保修 1 年，人为损坏不在范围内。",
}


def _search(keyword):
    """在知识库里检索关键词，返回命中的条目（工具：有明确动作/计算）。"""
    hits = {k: v for k, v in KB.items() if keyword in k or keyword in v}
    return {"hits": hits, "count": len(hits)}


def _calc_discount(price, rate):
    """按折扣率算最终价（工具：做计算）。"""
    return {"final": round(price * (1 - rate), 2)}


TOOLS = {
    "search_kb": {"description": "在客服知识库中检索关键词。",
                  "inputSchema": {"type": "object",
                                  "properties": {"keyword": {"type": "string"}},
                                  "required": ["keyword"]},
                  "func": _search},
    "calc_discount": {"description": "按折扣率计算最终价格。",
                      "inputSchema": {"type": "object",
                                      "properties": {"price": {"type": "number"},
                                                     "rate": {"type": "number"}},
                                      "required": ["price", "rate"]},
                      "func": _calc_discount},
}


def _answer_prompt(question="", evidence=""):
    return [
        {"role": "system", "content": "你是客服助手，只依据提供的政策条款回答，语气简洁专业。"},
        {"role": "user", "content": f"依据：{evidence}\n\n问题：{question}"},
    ]


PROMPTS = {
    "answer": {"description": "把检索到的政策条款 + 用户问题，组织成一次规范回答。",
               "arguments": [{"name": "question", "required": True},
                             {"name": "evidence", "required": False}],
               "func": _answer_prompt},
}


def handle_request(req):
    method, rid, params = req.get("method"), req.get("id"), req.get("params", {})
    ok = lambda r: {"jsonrpc": "2.0", "id": rid, "result": r}
    err = lambda c, m: {"jsonrpc": "2.0", "id": rid, "error": {"code": c, "message": m}}

    if method == "initialize":
        return ok({"protocolVersion": "2024-11-05",
                   "capabilities": {"tools": {}, "resources": {}, "prompts": {}},  # 三种都支持
                   "serverInfo": {"name": "support-assistant-server", "version": "1.0.0"}})

    if method == "tools/list":
        return ok({"tools": [{"name": n, "description": t["description"],
                              "inputSchema": t["inputSchema"]} for n, t in TOOLS.items()]})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments", {})
        if name not in TOOLS:
            return err(-32602, f"未知工具: {name}")
        try:
            return ok({"content": [{"type": "text", "text": json.dumps(TOOLS[name]["func"](**args),
                                                                       ensure_ascii=False)}],
                       "isError": False})
        except Exception as e:
            return ok({"content": [{"type": "text", "text": str(e)}], "isError": True})

    if method == "resources/list":
        return ok({"resources": [{"uri": f"policy://{k}", "name": k, "mimeType": "text/plain"}
                                 for k in KB]})
    if method == "resources/read":
        k = params.get("uri", "").removeprefix("policy://")
        if k not in KB:
            return err(-32602, f"资源不存在: {params.get('uri')}")
        return ok({"contents": [{"uri": f"policy://{k}", "mimeType": "text/plain", "text": KB[k]}]})

    if method == "prompts/list":
        return ok({"prompts": [{"name": n, "description": p["description"],
                                "arguments": p["arguments"]} for n, p in PROMPTS.items()]})
    if method == "prompts/get":
        name = params.get("name")
        if name not in PROMPTS:
            return err(-32602, f"提示不存在: {name}")
        msgs = PROMPTS[name]["func"](**params.get("arguments", {}))
        return ok({"description": PROMPTS[name]["description"],
                   "messages": [{"role": m["role"],
                                 "content": {"type": "text", "text": m["content"]}} for m in msgs]})

    return err(-32601, f"方法不存在: {method}")


def run_server():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        sys.stdout.write(json.dumps(handle_request(req)) + "\n")
        sys.stdout.flush()


# ============================== 客户端 / Agent 侧 ==============================
class MCPClient:
    def __init__(self, cmd):
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1)
        self._id = 0

    def call(self, method, params=None):
        self._id += 1
        self.proc.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        resp = json.loads(self.proc.stdout.readline())
        if "error" in resp:
            raise RuntimeError(resp["error"]["message"])
        return resp["result"]

    def close(self):
        self.proc.stdin.close(); self.proc.terminate()


# 关键词 → 知识库主题（模拟 LLM 的“意图识别”，真实里由 LLM 完成）
_TOPIC = {"退款": "refund", "refund": "refund", "配送": "shipping", "包邮": "shipping",
          "shipping": "shipping", "保修": "warranty", "warranty": "warranty"}


def answer_question(client, question):
    """Agent 单步问答的多能力编排：搜资源 → (可选)算折扣 → 套提示模板产出。"""
    # ① 用工具在知识库检索（工具内部读的是同一批“资源”数据）
    topic = next((t for kw, t in _TOPIC.items() if kw in question), None)
    evidence = ""
    if topic:
        hit = json.loads(client.call("tools/call",
                                     {"name": "search_kb", "arguments": {"keyword": topic}})["content"][0]["text"])
        evidence = " ".join(hit["hits"].values())

    # ② 若问题涉及“打折/优惠价”，再调计算工具（体现多步：检索之外还要算）
    nums = [float(x) for x in re.findall(r"\d+\.?\d*", question)]
    calc_note = ""
    if ("打" in question or "折" in question or "优惠" in question) and len(nums) >= 2:
        price = nums[0]
        # 中文“打8折”= 付原价的 80%，所以折扣率 rate = 1 - 8/10 = 0.2
        rate = round(1 - nums[1] / 10, 2)
        final = json.loads(client.call("tools/call",
                                       {"name": "calc_discount",
                                        "arguments": {"price": price, "rate": rate}})["content"][0]["text"])
        calc_note = f"（{price} 元按此折扣后为 {final['final']} 元）"

    # ③ 套用服务器提供的 answer 提示模板，产出规范回答的 messages（真实里喂给 LLM 生成终答）
    rendered = client.call("prompts/get",
                           {"name": "answer",
                            "arguments": {"question": question, "evidence": evidence or "（无匹配政策）"}})
    return evidence, calc_note, rendered["messages"]


def agent_loop():
    print("=" * 72)
    print(" 多能力 MCP Agent：一次发现 工具/资源/提示，按问题多步编排（检索→计算→套模板）")
    print("=" * 72)
    client = MCPClient([sys.executable, __file__, "--server"])
    try:
        info = client.call("initialize", {})
        caps = list(info["capabilities"])
        tools = [t["name"] for t in client.call("tools/list")["tools"]]
        resources = [r["uri"] for r in client.call("resources/list")["resources"]]
        prompts = [p["name"] for p in client.call("prompts/list")["prompts"]]
        print(f"\n① 能力发现：capabilities={caps}")
        print(f"   工具={tools}")
        print(f"   资源={resources}")
        print(f"   提示={prompts}")

        questions = ["你们的退款政策是什么？",
                     "一件 200 元的商品打8折是多少？保修多久？",
                     "今天天气怎么样？"]
        checks = []
        print("\n② 逐个问题走多步编排：")
        for q in questions:
            print(f"\n用户: {q}")
            evidence, calc_note, msgs = answer_question(client, q)
            final_user_msg = msgs[-1]["content"]["text"]
            print(f"  Agent 检索到依据: {evidence or '（无）'}")
            if calc_note:
                print(f"  Agent 计算得到: {calc_note}")
            print(f"  Agent 组装出待发给 LLM 的提示(节选): {final_user_msg[:80].replace(chr(10), ' ')}...")
            checks.append((q, evidence, calc_note))

        # ---- 自检：验证多步编排的每一环都真的生效 ----
        assert checks[0][1] and "退款" in checks[0][1]                 # 退款问题检索到退款政策
        assert checks[1][2] and "160.0" in checks[1][2]               # 200 打 8 折 = 160
        assert checks[1][1] and "保修" in checks[1][1]                 # 同一问题也检索到保修政策
        assert not checks[2][1]                                       # 天气问题无匹配政策(诚实返回空)

        print("\n" + "=" * 72)
        print(" ✅ 自检通过：Agent 一次发现三种原语；")
        print("    · 退款问题 → 命中资源检索；")
        print("    · 打折+保修问题 → 同时触发【检索 + 计算(200×0.8=160)】两步工具；")
        print("    · 无关问题 → 诚实返回“无匹配政策”，不编造。")
        print(" 这就是 MCP 让 AI 会“组合使用多种能力”的完整闭环。")
        print("=" * 72)
    finally:
        client.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        run_server()
    else:
        agent_loop()
