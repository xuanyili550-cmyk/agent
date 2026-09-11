"""
================================================================================
 MCP Course · Chapter 3 · 案例：知识库服务器——资源(Resources) + 提示(Prompts) 实战
================================================================================
 chapter3 的笔记讲清了资源/提示的协议消息。本案例做一个更“像真项目”的东西：
 一个【个人知识库 MCP 服务器】，把它当成给 AI 用的一个小型笔记本：

   · 资源(Resources)：把每条笔记暴露成一个只读资源，用 URI 标识(note://python-gil 等)。
     - resources/list          列出所有笔记
     - resources/templates/list 声明“URI 模板” note://{id}，告诉客户端可按 id 取任意笔记
     - resources/read          按 URI 读一条笔记的正文
     为什么用资源而不是工具？因为读笔记是【只读、无副作用】的取上下文操作 —— 这正是资源的定位。

   · 提示(Prompts)：预定义两个可复用的工作流模板，客户端填参数即可拿到渲染好的 messages：
     - summarize(topic)         让 AI 用知识库内容总结某主题
     - qa(question)             基于知识库回答问题
     为什么用提示？把“怎么问 AI”固化成模板，保证团队每次交互一致、可复用。

 关键点(贯穿全章)：资源提供“读什么”，提示提供“怎么用”，工具提供“做什么”。三者配合，
 AI 就能【先读资源拿上下文 → 套提示模板 → 生成回答】。传输仍是真实 stdio + JSON-RPC 2.0。

 运行：
   python3 chapter3_资源与提示实战案例.py           # 客户端：发现并使用 资源 + 提示
   python3 chapter3_资源与提示实战案例.py --server   # 仅作为 MCP 服务器运行
================================================================================
"""

import json
import subprocess
import sys

# ---- 知识库数据：每条笔记 = 一个只读资源。真实项目里这可能来自数据库/文件系统 ----
NOTES = {
    "python-gil": {"title": "Python GIL", "tags": ["python", "concurrency"],
                   "body": "GIL 是全局解释器锁，同一时刻只有一个线程执行字节码；"
                           "CPU 密集用多进程，IO 密集用多线程/异步。"},
    "mcp-basics": {"title": "MCP 基础", "tags": ["mcp", "protocol"],
                   "body": "MCP 用 JSON-RPC 2.0 定义 AI 与外部能力的通信，四大原语："
                           "工具/资源/提示/采样，传输可走 stdio 或 HTTP+SSE。"},
    "vector-db": {"title": "向量数据库", "tags": ["rag", "embedding"],
                  "body": "向量库存 embedding 并做近邻检索，是 RAG 的召回层，"
                          "常见实现有 FAISS / Milvus / pgvector。"},
}


def _uri(note_id):
    return f"note://{note_id}"


# ---- 提示模板：填参数后返回一组 messages（客户端可直接喂给 LLM）----
def _summarize_prompt(topic=""):
    # 把知识库里相关笔记拼进上下文，再让 AI 总结 —— 演示“资源 + 提示”如何协作
    context = "\n".join(f"- {n['title']}: {n['body']}" for n in NOTES.values()
                        if topic.lower() in (n["title"] + n["body"]).lower())
    return [
        {"role": "system", "content": "你是一个基于给定知识库作答的助手，只用提供的资料。"},
        {"role": "user", "content": f"请根据以下知识库内容，总结「{topic}」：\n{context or '（无相关内容）'}"},
    ]


def _qa_prompt(question=""):
    return [
        {"role": "system", "content": "你是知识库问答助手，答不出就说不知道，不要编造。"},
        {"role": "user", "content": f"问题：{question}"},
    ]


PROMPTS = {
    "summarize": {"description": "基于知识库总结某个主题。",
                  "arguments": [{"name": "topic", "description": "要总结的主题", "required": True}],
                  "func": _summarize_prompt},
    "qa": {"description": "基于知识库回答一个问题。",
           "arguments": [{"name": "question", "description": "用户问题", "required": True}],
           "func": _qa_prompt},
}


# ==============================================================================
# 服务器：分发 资源 + 提示 相关的 JSON-RPC 方法
# ==============================================================================
def handle_request(req):
    method, rid, params = req.get("method"), req.get("id"), req.get("params", {})
    ok = lambda r: {"jsonrpc": "2.0", "id": rid, "result": r}
    err = lambda c, m: {"jsonrpc": "2.0", "id": rid, "error": {"code": c, "message": m}}

    if method == "initialize":
        return ok({"protocolVersion": "2024-11-05",
                   "capabilities": {"resources": {}, "prompts": {}},   # 声明支持 资源 + 提示
                   "serverInfo": {"name": "knowledge-base-server", "version": "1.0.0"}})

    # ---- 资源 ----
    if method == "resources/list":
        return ok({"resources": [
            {"uri": _uri(nid), "name": n["title"], "mimeType": "text/plain",
             "description": "标签: " + ", ".join(n["tags"])}
            for nid, n in NOTES.items()]})

    if method == "resources/templates/list":
        # URI 模板：告诉客户端“凡是 note://{id} 都能读”，不必逐个枚举
        return ok({"resourceTemplates": [
            {"uriTemplate": "note://{id}", "name": "笔记", "mimeType": "text/plain",
             "description": "按笔记 id 读取正文"}]})

    if method == "resources/read":
        uri = params.get("uri", "")
        nid = uri.removeprefix("note://")
        if nid not in NOTES:
            return err(-32602, f"资源不存在: {uri}")
        n = NOTES[nid]
        return ok({"contents": [{"uri": uri, "mimeType": "text/plain",
                                 "text": f"{n['title']}\n{n['body']}"}]})

    # ---- 提示 ----
    if method == "prompts/list":
        return ok({"prompts": [{"name": p_name, "description": p["description"],
                                "arguments": p["arguments"]} for p_name, p in PROMPTS.items()]})

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


# ==============================================================================
# 客户端：发现并使用 资源 + 提示，演示“读资源 → 套提示”的协作
# ==============================================================================
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


def run_client():
    print("=" * 72)
    print(" 知识库 MCP 服务器：资源(只读取上下文) + 提示(可复用模板) 实战")
    print("=" * 72)
    c = MCPClient([sys.executable, __file__, "--server"])
    try:
        info = c.call("initialize", {})
        print(f"\n握手成功，服务器能力 = {list(info['capabilities'])}")

        # ① 资源：列出 → 看模板 → 读一条
        res = c.call("resources/list")["resources"]
        print(f"\n① resources/list 发现 {len(res)} 条笔记(资源)：")
        for r in res:
            print(f"   · {r['uri']}  [{r['name']}]  {r['description']}")

        tmpl = c.call("resources/templates/list")["resourceTemplates"]
        print(f"\n   resources/templates/list → URI 模板: {tmpl[0]['uriTemplate']}"
              f"（凡符合此模板的 URI 都能读）")

        one = c.call("resources/read", {"uri": "note://mcp-basics"})["contents"][0]
        print("\n   resources/read note://mcp-basics →")
        print("     " + one["text"].replace("\n", "\n     "))

        # ② 提示：发现 → 渲染 summarize（内部会自动把相关资源拼进上下文）
        prompts = c.call("prompts/list")["prompts"]
        print(f"\n② prompts/list 发现 {len(prompts)} 个提示：")
        for p in prompts:
            print(f"   · {p['name']}: {p['description']}  参数={[a['name'] for a in p['arguments']]}")

        got = c.call("prompts/get", {"name": "summarize", "arguments": {"topic": "MCP"}})
        print("\n   prompts/get summarize(topic='MCP') → 渲染出", len(got["messages"]), "条 messages：")
        for m in got["messages"]:
            print(f"     [{m['role']}] {m['content']['text'][:70].replace(chr(10), ' ')}...")

        # ③ 边界：读不存在的资源要报错
        print("\n③ 边界：读不存在的资源 →", end=" ")
        try:
            c.call("resources/read", {"uri": "note://not-exist"})
        except RuntimeError as ex:
            print("服务器正确报错:", ex)

        # ---- 自检 ----
        assert len(res) == 3
        assert "MCP" in one["text"]
        assert len(got["messages"]) == 2
        assert "MCP" in got["messages"][1]["content"]["text"]  # summarize 把 MCP 相关笔记拼进去了
        print("\n" + "=" * 72)
        print(" ✅ 自检通过：资源(列出/模板/读取/越界报错) + 提示(发现/带参渲染) 全走通；")
        print("    并演示了“读资源拿上下文 → 套提示模板”的真实协作。都是标准 JSON-RPC 消息。")
        print("=" * 72)
    finally:
        c.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        run_server()
    else:
        run_client()
