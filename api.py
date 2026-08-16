# -*- coding:utf-8 -*-
"""
FastAPI 服务入口。

接口：
- GET  /                 健康检查 + 简易首页
- GET  /api/health        健康检查
- POST /api/chat          问答接口（SSE 流式）
- POST /api/chat/sync     问答接口（非流式，返回完整 JSON）
- POST /api/chat/clear    清除指定会话历史

启动：
    python

    api.py
    或
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import sys
import json
import asyncio
from collections import defaultdict

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel
from base import logger, Config

conf = Config()

# 延迟初始化系统（避免 import 时就连接 Milvus/Ollama）
_system = None


def get_system():
    global _system
    if _system is None:
        from main import CampusQASystem
        _system = CampusQASystem()
    return _system


app = FastAPI(title="理工学生手册问答系统", version="1.0.0")

# 简易会话历史（内存，按 session_id 隔离；生产环境建议换 MySQL/Redis）
_session_history = defaultdict(list)
MAX_HISTORY = 10  # 每个会话保留最近 10 轮

# 请求超时时间（秒）
REQUEST_TIMEOUT = 60

# 兜底策略提示语
FALLBACK_MESSAGE = (
    "系统响应超时，已自动结束本次请求。"
    f"如需帮助，请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
)


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"


class ClearRequest(BaseModel):
    session_id: str = "default"


#
# @app.get("/", response_class=HTMLResponse)
# async def index():
#     """简易首页。"""
#     return """
#     <html><head><meta charset='utf-8'><title>校园学生手册问答系统</title></head>
#     <body style='font-family:sans-serif;max-width:800px;margin:40px auto;padding:20px'>
#       <h2>🎓 校园学生手册智能问答系统</h2>
#       <p>基于 Ollama + BERT + RAG 的三意图问答系统。</p>
#       <h3>接口</h3>
#       <ul>
#         <li><code>POST /api/chat</code> SSE 流式问答</li>
#         <li><code>POST /api/chat/sync</code> 非流式问答</li>
#         <li><code>GET /api/health</code> 健康检查</li>
#       </ul>
#       <h3>示例</h3>
#       <pre>curl -N -X POST http://localhost:8000/api/chat \\
#   -H "Content-Type: application/json" \\
#   -d '{"query":"晚归怎么处分","session_id":"test"}'</pre>
#     </body></html>
#     """


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "campus_qa", "version": "1.0.0"}


@app.post("/api/chat/clear")
async def clear_history(req: ClearRequest):
    """清除指定会话的历史记录（新开对话框）。"""
    if req.session_id in _session_history:
        _session_history[req.session_id].clear()
        logger.info(f"会话 {req.session_id} 历史已清除")
    return {"status": "ok", "message": "历史记录已清除", "session_id": req.session_id}


def _save_history(session_id: str, question: str, answer: str):
    """保存对话历史，保留最近 MAX_HISTORY 条。"""
    _session_history[session_id].append(
        {"question": question, "answer": answer}
    )
    if len(_session_history[session_id]) > MAX_HISTORY:
        _session_history[session_id] = _session_history[session_id][-MAX_HISTORY:]


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """SSE 流式问答，支持 60 秒超时兜底。"""
    system = get_system()
    history = _session_history.get(req.session_id, [])

    async def event_stream():
        full_answer = []
        start_time = asyncio.get_event_loop().time()
        timed_out = False

        try:
            for chunk in system.stream_answer(req.query, history):
                # 检查是否已超时
                if asyncio.get_event_loop().time() - start_time > REQUEST_TIMEOUT:
                    timed_out = True
                    logger.warning(f"会话 {req.session_id} 请求超时（>{REQUEST_TIMEOUT}s），触发兜底")
                    yield f"data: {json.dumps({'type':'token','content':FALLBACK_MESSAGE}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type':'done'}, ensure_ascii=False)}\n\n"
                    break

                if chunk.startswith("[intent:"):
                    intent = chunk.replace("[intent:", "").replace("]", "")
                    yield f"data: {json.dumps({'type':'intent','intent':intent}, ensure_ascii=False)}\n\n"
                    continue
                full_answer.append(chunk)
                yield f"data: {json.dumps({'type':'token','content':chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)  # 让出事件循环

            if not timed_out:
                yield f"data: {json.dumps({'type':'done'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"SSE 流式出错: {e}")
            yield f"data: {json.dumps({'type':'error','message':str(e)}, ensure_ascii=False)}\n\n"

        # 保存历史（超时情况下也保存已生成的部分）
        answer_str = "".join(full_answer)
        if timed_out:
            answer_str = FALLBACK_MESSAGE
        _save_history(req.session_id, req.query, answer_str)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/chat/sync")
async def chat_sync(req: ChatRequest):
    """非流式问答，返回完整 JSON，支持 60 秒超时兜底。"""
    system = get_system()
    history = _session_history.get(req.session_id, [])

    try:
        answer = await asyncio.wait_for(
            asyncio.to_thread(system.answer, req.query, history),
            timeout=REQUEST_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.warning(f"会话 {req.session_id} 同步请求超时（>{REQUEST_TIMEOUT}s），触发兜底")
        answer = FALLBACK_MESSAGE

    # 保存历史
    _save_history(req.session_id, req.query, answer)
    return JSONResponse({"query": req.query, "answer": answer})




@app.get("/", response_class=HTMLResponse)
async def index():
    """返回聊天前端页面。"""
    index_path = os.path.join(_project_root, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>理工学生手册问答系统</h1><p>index.html 不存在</p>"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)