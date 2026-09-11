"""
================================================================================
 MCP 实战 · 案例3 · 把小 LLM 文本生成(Ch11)暴露成 MCP 工具 generate
================================================================================
 为什么用 MCP 包 LLM：Host 侧的主模型可能不擅长/不方便直接跑某个本地小模型。把一个
 本地部署的小 LLM(SmolLM2-135M-Instruct)包成 MCP 工具 generate，Agent 就能把"生成"这个
 动作外包给它——比如离线/隐私场景用本地小模型草拟文本，主模型只做编排。这体现 MCP 让
 "能力"跨模型、跨进程复用：谁部署的模型不重要，会说 JSON-RPC 就能被调用。

 生成原理(Ch11)：用 chat template 组织对话 → 模型自回归 generate → 解码新增 token。
 小模型 + max_new_tokens 设小(20~40) 保证本机能在合理时间跑完；生成本身较慢属正常。

 手写 MCP：纯 Python stdio + JSON-RPC 2.0(照 chapter2)。模型惰性加载 + 单例。
 用法：
   python3 案例3_MCP服务器_LLM生成.py            # 客户端(自动 spawn 服务器)走完整流程
   python3 案例3_MCP服务器_LLM生成.py --server    # 只作为 MCP 服务器

 [Ch11] HuggingFaceTB/SmolLM2-135M-Instruct
================================================================================
"""
import json
import subprocess
import sys

_STATE = {"dev": None, "tok": None, "model": None}


def _pick_device():
    import torch
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def _ensure_model():
    if _STATE["model"] is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        _STATE["dev"] = _pick_device()
        name = "HuggingFaceTB/SmolLM2-135M-Instruct"
        _STATE["tok"] = AutoTokenizer.from_pretrained(name)
        _STATE["model"] = AutoModelForCausalLM.from_pretrained(name).to(_STATE["dev"]).eval()


def _generate(prompt: str, max_new_tokens: int = 20):
    """[Ch11] chat template → 自回归 generate → 只解码新增 token。"""
    import torch
    _ensure_model()
    tok, model = _STATE["tok"], _STATE["model"]
    max_new_tokens = max(1, min(int(max_new_tokens), 80))   # 上限保护，防生成过久
    # 用 chat template 把用户消息包成 Instruct 模型期望的对话格式(先渲染成字符串再分词，最稳)
    messages = [{"role": "user", "content": prompt}]
    text_in = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = tok(text_in, return_tensors="pt").to(_STATE["dev"])
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,   # 贪心=可复现
            pad_token_id=tok.eos_token_id)
    new_tokens = out[0][enc["input_ids"].shape[1]:]   # 只取新增部分，去掉输入回显
    text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    return {"prompt": prompt, "completion": text, "new_tokens": int(new_tokens.shape[0])}


# ==============================================================================
# MCP 服务器
# ==============================================================================
TOOLS = {
    "generate": {
        "description": "用本地小 LLM(SmolLM2-135M-Instruct)根据提示生成一小段文本。适合离线/隐私场景草拟内容。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "给模型的指令/提示"},
                "max_new_tokens": {"type": "integer", "description": "最多生成多少新 token(默认20，上限80)", "default": 20},
            },
            "required": ["prompt"],
        },
        "func": _generate,
    },
}


def handle_request(req: dict) -> dict:
    method, req_id = req.get("method"), req.get("id")
    params = req.get("params", {})

    def ok(result): return {"jsonrpc": "2.0", "id": req_id, "result": result}
    def err(code, msg): return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": msg}}

    if method == "initialize":
        return ok({"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                   "serverInfo": {"name": "llm-generate-server", "version": "1.0.0"}})
    if method == "tools/list":
        return ok({"tools": [
            {"name": n, "description": t["description"], "inputSchema": t["inputSchema"]}
            for n, t in TOOLS.items()]})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments", {})
        if name not in TOOLS:
            return err(-32602, f"Unknown tool: {name}")
        try:
            result = TOOLS[name]["func"](**args)
            return ok({"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                       "isError": False})
        except Exception as e:
            return ok({"content": [{"type": "text", "text": str(e)}], "isError": True})
    return err(-32601, f"Method not found: {method}")


def run_server():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        sys.stdout.write(json.dumps(handle_request(req), ensure_ascii=False) + "\n")
        sys.stdout.flush()


# ==============================================================================
# MCP 客户端
# ==============================================================================
class MCPClient:
    def __init__(self, server_cmd):
        self.proc = subprocess.Popen(server_cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        self._id = 0

    def call(self, method, params=None):
        self._id += 1
        req = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        resp = json.loads(self.proc.stdout.readline())
        if "error" in resp:
            raise RuntimeError(resp["error"]["message"])
        return resp["result"]

    def close(self):
        self.proc.stdin.close()
        self.proc.terminate()


def run_client():
    # print("=" * 72 + "\n MCP 案例3：把小 LLM 生成(Ch11)暴露成工具 generate\n" + "=" * 72)
    client = MCPClient([sys.executable, __file__, "--server"])
    try:
        info = client.call("initialize", {"protocolVersion": "2024-11-05",
                                          "capabilities": {}, "clientInfo": {"name": "gen-client"}})
        # print(f"\n① initialize → 服务器={info['serverInfo']['name']} v{info['serverInfo']['version']}")

        tools = client.call("tools/list")["tools"]
        # print(f"\n② tools/list 发现工具：{[t['name'] for t in tools]}")   # 自检: 发现 generate

        # print("\n③ tools/call 调用本地小 LLM 生成(首次加载+生成较慢，请耐心)：")
        prompts = [
            "Give one short tip to write a good git commit message.",
            "Suggest a name for a friendly coffee shop.",
        ]
        got = None
        for p in prompts:
            r = client.call("tools/call", {"name": "generate",
                                           "arguments": {"prompt": p, "max_new_tokens": 24}})
            data = json.loads(r["content"][0]["text"])
            got = got or data
            # print(f"\n   Prompt: {p}")
            # print(f"   → ({data['new_tokens']} toks) {data['completion']}")   # 自检: new_tokens>0 且有文本

        assert got["new_tokens"] > 0 and len(got["completion"]) > 0
        print("\n✅ 自检通过：MCP 把本地小 LLM 的生成能力暴露成工具并真实产出文本(new_tokens>0)。")
    finally:
        client.close()


# ==============================================================================
# 生产要点(上云怎么搭)
# ==============================================================================
# · 真实生产不用 transformers.generate 扛并发，改用推理服务器 vLLM / TGI(连续批处理 + PagedAttention)，
#   MCP 工具内部只做 HTTP 调用；或直接把 OpenAI/Anthropic 兼容端点包成工具。
# · 流式：长生成要流式返回(SSE)，别等全部生成完；MCP 用 Streamable HTTP 传输可承载流式(见案例5)。
# · 采样参数下放：temperature/top_p/max_tokens 做成 inputSchema 参数，Host 可控;贪心(do_sample=False)可复现。
# · 超时与配额：generate 可能很慢，要设超时、限流、单次 token 上限，防止一个请求拖垮服务。
# · 安全：对 prompt 做注入/越权过滤;生成结果落地前审计(尤其被下游当代码/命令执行时)。

# ==============================================================================
# 面试题(MCP + LLM)
# ==============================================================================
# Q: 都能调 LLM 了，MCP 和 OpenAI/Anthropic 的 function calling 是什么关系？
# A: 互补，不是二选一。function calling 是"模型决定调哪个工具、生成什么参数"的能力(模型侧)；MCP 是"工具
#    怎么被描述、发现、连接、调用"的标准协议(工具侧)。典型链路：MCP 提供 tools/list → 塞给 LLM 做
#    function calling 选工具 → Host 通过 MCP tools/call 执行 → 结果回喂 LLM。见案例4编排。
# Q: 为什么需要 MCP，直接给每个模型写工具插件不行吗？
# A: 组合爆炸——M 个 AI 应用 × N 个工具 = M×N 套定制集成。MCP 定标准协议后变 M+N：工具实现一次(MCP
#    Server)，任何支持 MCP 的 Host 都能用，反之亦然。这就是"AI 的 USB-C"。
# Q: MCP 的 Sampling 原语是什么？和本例的 generate 有何不同?
# A: 本例 generate 是"服务器自己有个模型来生成"。Sampling 是反向：MCP 服务器请求 Host 用它的 LLM 来生成
#    (服务器不自带模型，借 Host 的)。适合服务器逻辑需要 LLM 推理但不想自带模型的场景。
# Q: 生成类工具在生产上最大的坑?
# A: 延迟与并发。单请求慢、且 GPU 显存有限，必须上连续批处理(vLLM/TGI)+ 流式 + 超时限流，否则高并发直接雪崩。

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        run_server()
    else:
        run_client()
