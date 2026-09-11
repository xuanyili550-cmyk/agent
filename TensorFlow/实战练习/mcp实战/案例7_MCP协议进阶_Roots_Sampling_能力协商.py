"""
================================================================================
 MCP 实战 · 案例7 · 协议进阶：能力协商 / initialized 通知 / Roots / Sampling / 资源模板
================================================================================
 案例1-6 覆盖了 Tools/Resources/Prompts 主干；本文件补上 MCP 协议里更“底层/进阶”的部分，
 都用【手写 JSON-RPC 2.0 消息】演示（这些特性涉及 server↔client 双向或 Host 的 LLM，
 用单进程构造并校验真实消息形态最清楚、也能直接跑）：
   ① 能力协商取交集：initialize 时客户端和服务器各报能力，只有【双方都支持】的才可用。
   ② initialized 通知：握手完成后客户端发一条【无 id 的通知】(notification)，区别于请求。
   ③ Roots(根)：客户端把“文件系统根/工作目录”暴露给服务器 —— 服务器反向发 roots/list 请求。
   ④ Sampling(采样)：服务器【反过来请求 Host 的 LLM】生成 —— sampling/createMessage 消息。
   ⑤ 资源模板 resources/templates/list：带参数的 URI 模板(如 note://{id})。
 跑：python3 案例7_MCP协议进阶_Roots_Sampling_能力协商.py
================================================================================
"""
import json


def show(title, msg):
    print(f"  {title}:\n    " + json.dumps(msg, ensure_ascii=False))


# ==============================================================================
# ① 能力协商取交集：只有双方都支持的能力才能用
# ==============================================================================
def capability_negotiation():
    # print("=" * 72, "\n① 能力协商取交集(capability negotiation)\n" + "=" * 72)
    client_caps = {"roots": {}, "sampling": {}}                       # 客户端(Host)支持
    server_caps = {"tools": {}, "resources": {}, "sampling": {}}      # 服务器支持
    # initialize 请求(客户端→服务器)带上自己的能力
    init_req = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": client_caps,
                           "clientInfo": {"name": "demo-host", "version": "1.0"}}}
    show("initialize 请求(带客户端能力)", init_req)
    # 服务器回自己的能力
    init_resp = {"jsonrpc": "2.0", "id": 1, "result": {
        "protocolVersion": "2024-11-05", "capabilities": server_caps,
        "serverInfo": {"name": "demo-server", "version": "1.0"}}}
    show("initialize 响应(带服务器能力)", init_resp)
    # 取交集：双方都声明的能力才真正可用
    usable = set(client_caps) & set(server_caps)
    print(f"  ⇒ 客户端={set(client_caps)}  服务器={set(server_caps)}  可用交集={usable}")
    assert usable == {"sampling"}                                     # 只有 sampling 双方都有
    # print("  所以本会话只有 sampling 能用；roots 服务器不认、tools/resources 客户端不主动用。")


# ==============================================================================
# ② initialized 通知：无 id 的通知消息(不是请求)
# ==============================================================================
def initialized_notification():
    # print("\n" + "=" * 72, "\n② initialized 通知(无 id = 通知，不需要响应)\n" + "=" * 72)
    note = {"jsonrpc": "2.0", "method": "notifications/initialized"}  # ★没有 "id" 字段
    show("客户端握手完成后发", note)
    def is_notification(m):
        return "method" in m and "id" not in m                       # 区分四类消息的关键
    assert is_notification(note)
    assert not is_notification({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    # print("  四类消息：请求(有id+method)/响应(有id+result或error)/通知(有method无id)/错误(有id+error)。")
    # print("  通知是“单向告知、不等回复”，如 initialized、progress、cancelled、日志。")


# ==============================================================================
# ③ Roots：客户端暴露根，服务器反向请求 roots/list
# ==============================================================================
def roots():
    # print("\n" + "=" * 72, "\n③ Roots(客户端暴露文件系统根，服务器反向请求)\n" + "=" * 72)
    # 注意方向：这是【服务器→客户端】的请求(和 tools/call 方向相反)
    req = {"jsonrpc": "2.0", "id": 100, "method": "roots/list"}
    show("服务器→客户端 请求", req)
    resp = {"jsonrpc": "2.0", "id": 100, "result": {"roots": [
        {"uri": "file:///Users/me/project", "name": "我的项目"},
        {"uri": "file:///Users/me/data", "name": "数据目录"}]}}
    show("客户端→服务器 响应", resp)
    # print("  用途：让服务器知道“它被允许在哪些目录/根下工作”(安全边界)。需要客户端声明 roots 能力。")
    assert resp["result"]["roots"][0]["uri"].startswith("file://")


# ==============================================================================
# ④ Sampling：服务器反过来请求 Host 的 LLM 生成
# ==============================================================================
def sampling():
    # print("\n" + "=" * 72, "\n④ Sampling(服务器反请 Host 的 LLM 生成)\n" + "=" * 72)
    # 方向：服务器→客户端(Host)。服务器自己没有 LLM，借 Host 的
    req = {"jsonrpc": "2.0", "id": 200, "method": "sampling/createMessage", "params": {
        "messages": [{"role": "user", "content": {"type": "text", "text": "用一句话总结 MCP"}}],
        "systemPrompt": "你是简洁的助手。", "maxTokens": 100, "includeContext": "none"}}
    show("服务器→客户端 sampling/createMessage", req)
    resp = {"jsonrpc": "2.0", "id": 200, "result": {
        "role": "assistant", "model": "claude-3-5-sonnet",
        "content": {"type": "text", "text": "MCP 是让 AI 应用连接外部工具/数据的开放协议。"},
        "stopReason": "endTurn"}}
    show("客户端→服务器 响应(Host 的 LLM 生成结果)", resp)
    # print("  价值：工具服务器无需自带模型，也能用上 Host 的 LLM 做智能处理(如总结检索结果)；")
    # print("  Host 通常会加“人在环”确认，防止服务器滥用 LLM。需要客户端声明 sampling 能力。")
    assert req["params"]["maxTokens"] == 100


# ==============================================================================
# ⑤ 资源模板 resources/templates/list：带参数的 URI 模板
# ==============================================================================
def resource_templates():
    # print("\n" + "=" * 72, "\n⑤ 资源模板 resources/templates/list(带参 URI)\n" + "=" * 72)
    req = {"jsonrpc": "2.0", "id": 300, "method": "resources/templates/list"}
    resp = {"jsonrpc": "2.0", "id": 300, "result": {"resourceTemplates": [
        {"uriTemplate": "note://{id}", "name": "笔记", "description": "按 id 读取笔记"},
        {"uriTemplate": "user://{name}/profile", "name": "用户资料"}]}}
    show("请求", req)
    show("响应(模板列表)", resp)
    # 客户端按模板填参，再 resources/read 具体 URI
    read = {"jsonrpc": "2.0", "id": 301, "method": "resources/read", "params": {"uri": "note://42"}}
    show("客户端按模板填好参数后 read", read)
    # print("  普通 resources/list 是“固定 URI 列表”；模板是“URI 带参数”，适合“按 id/名字取一类资源”。")


if __name__ == "__main__":
    capability_negotiation()
    initialized_notification()
    roots()
    sampling()
    resource_templates()
    # print("\n" + "=" * 72)
    print("✅ MCP 协议进阶跑通：能力协商取交集 → initialized 通知 → Roots → Sampling → 资源模板。")
    # print("面试：Q 能力协商为什么取交集? Q 通知和请求怎么区分? Q Roots/Sampling 方向和普通调用有何不同?")
    # print("     Q Sampling 有什么用/风险? Q 资源模板 vs 普通资源? (见 README_mcp实战导航.py)")
