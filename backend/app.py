"""
知识库问答 FastAPI 服务

API 接口：
  GET  /api/status   — 获取已加载文件列表
  POST /api/chat     — 发送聊天消息
  POST /api/load     — 按文件路径加载文档
  POST /api/remove   — 按文件路径删除文档
  POST /api/upload   — 上传文件并加载到知识库
  GET  /             — 前端页面
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

load_dotenv()

from agent_core import (
    run_query,
    do_load_document,
    do_remove_document,
)

from database import init_db, get_all_files


#启动时只初始化表结构，不管增删改
@asynccontextmanager
async def lifespan(app: FastAPI):
    #启动时执行
    await init_db()
    print(f"数据库已连接")
    yield
    #关闭时执行（如果有需要关闭的连接池放这里）
    print(f"服务正常关闭")


app = FastAPI(title="知识库问答 Agent",lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatReq(BaseModel):
    message: str
    session_id: str = "default"


class FileReq(BaseModel):
    filepath: str


#查看文件列表
@app.get("/api/status")
async def get_status():
    files = await get_all_files()
    return {"loaded_files": files}


@app.post("/api/chat")
async def chat(req: ChatReq):
    """发送消息给 Agent 并获取回答。"""
    answer = await asyncio.to_thread(run_query, req.message,req.session_id)
    return {"answer": answer}


@app.post("/api/load")
async def load_file(req: FileReq):
    """按文件路径加载文档到知识库。"""
    result = await asyncio.to_thread(do_load_document, req.filepath)
    return {"result": result}


@app.post("/api/remove")
async def remove_file(req: FileReq):
    result = await asyncio.to_thread(do_remove_document, req.filepath)
    return {"result": result}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件到服务器并加载到知识库。"""
    try:
        upload_dir = Path(__file__).resolve().parent / "uploaded_files"
        upload_dir.mkdir(exist_ok=True)

        raw_name = file.filename or "uploaded_file"
        safe_name = Path(raw_name).name
        filepath = upload_dir / safe_name

        counter = 1
        while filepath.exists():
            stem = filepath.stem
            suffix = filepath.suffix
            filepath = upload_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)

        result = await asyncio.to_thread(do_load_document,str(filepath))
        return {"result": result, "filepath": str(filepath)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def serve_frontend():
    """提供前端页面。浏览器打开 http://localhost:8000 即可使用。"""
    frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    if not frontend_path.exists():
        return HTMLResponse("<h1>前端文件未找到</h1>")
    html = frontend_path.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
