from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..services import stream
router = APIRouter(tags=["chat"])
@router.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()           # {"message": "..."}
            try:
                for tok in stream.stream_reply(data.get("message", "")):
                    await ws.send_json({"token": tok, "done": False})
                await ws.send_json({"token": "", "done": True})
            except Exception as e:
                await ws.send_json({"error": str(e), "done": True})
    except WebSocketDisconnect:
        return
