"""
作品集项目1 · 中文情感分析 API + Web 界面（生产级，可部署 HF Spaces）
--------------------------------------------------------------------------------
 一句话：把一个中文情感分类模型包成【带 Web 界面 + REST API】的在线服务。
 生产要点：模型单例加载(冷启动一次)、输入校验(gr.Error)、批量、队列并发、FastAPI 挂载出 API、
          示例、可观测(每次请求打日志)。中文文本用中文模型(英文模型喂中文会失准)。
 运行：
   python3 app.py smoke   # 只构建 + 调函数自检(不起服务，CI/本地验证用)
   python3 app.py         # 起 Gradio 网页(HF Spaces 默认入口)
   python3 app.py api     # 起 FastAPI(网页挂 /，API 在 /api/predict)  uvicorn 亦可
"""
import sys
import gradio as gr

MODEL = "uer/roberta-base-finetuned-dianping-chinese"   # 中文点评情感(0=负面 1=正面)
_STATE = {}


def _load():
    """模型单例：第一次调用才加载并常驻(生产避免每请求重载)。"""
    if "model" not in _STATE:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _STATE["dev"] = "mps" if torch.backends.mps.is_available() else "cpu"
        _STATE["tok"] = AutoTokenizer.from_pretrained(MODEL)
        _STATE["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL).to(_STATE["dev"]).eval()
    return _STATE


def predict(text: str) -> dict:
    """核心推理：中文文本 → {正面概率, 负面概率}。供 Web 和 API 共用。"""
    import torch
    if not text or not text.strip():
        raise gr.Error("请输入要分析的中文文本")          # 输入校验：前端弹友好红条
    s = _load()
    enc = s["tok"](text[:512], return_tensors="pt", truncation=True).to(s["dev"])
    with torch.no_grad():
        # softmax：把模型输出的 logits(未归一化分数)变成两类的概率，和为1
        probs = torch.softmax(s["model"](**enc).logits, -1)[0]
    return {"正面": float(probs[1]), "负面": float(probs[0])}


def build_demo():
    """Gradio 界面(Gradio 6.x API)。analytics_enabled 放构造器；不用已废弃的 allow_flagging。"""
    return gr.Interface(
        fn=predict,
        inputs=gr.Textbox(label="中文文本", lines=3, placeholder="例如：这家店服务很好，菜也好吃！"),
        outputs=gr.Label(label="情感", num_top_classes=2),
        title="中文情感分析",
        description="输入中文评论/文本，判断情感极性(正面/负面)。模型：uer/roberta 点评微调。",
        examples=[["这家店服务很好，下次还来！"], ["质量太差了，用一次就坏，退货！"],
                  ["还行吧，没什么特别的感觉。"]],
        flagging_mode="never",
        analytics_enabled=False,
    )


def make_api():
    """FastAPI：网页挂在 /，REST API 在 /api/predict(生产给别的服务调用)。"""
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="中文情感分析 API")

    class Req(BaseModel):
        text: str

    @app.post("/api/predict")
    def api_predict(req: Req):
        try:
            return {"ok": True, "result": predict(req.text)}
        except gr.Error as e:
            return {"ok": False, "error": str(e)}

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    return gr.mount_gradio_app(app, build_demo().queue(), path="/")   # 队列并发 + 挂载


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if arg == "smoke":
        build_demo()                                        # 能构建
        r = predict("这家餐厅太棒了，强烈推荐！")
        print("predict('这家餐厅太棒了') →", {k: round(v, 3) for k, v in r.items()})
        assert r["正面"] > r["负面"]
        assert predict("垃圾产品，再也不买")["负面"] > 0.5
        try:
            predict("   ")
        except gr.Error:
            print("空输入校验 → 正确抛 gr.Error ✅")
        print("✅ 项目1 自检通过：中文情感 predict + Web 构建 + 输入校验。")
    elif arg == "api":
        import uvicorn
        uvicorn.run(make_api(), host="0.0.0.0", port=7860)  # 网页 http://localhost:7860 ，API /api/predict
    else:
        build_demo().queue().launch()                        # HF Spaces 默认入口
