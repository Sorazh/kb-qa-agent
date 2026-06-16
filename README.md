---
title: 知识库问答 Agent
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---
# 知识库问答 Agent (KB QA Agent)

基于 **LangChain v1 + FastAPI** 的知识库问答系统。上传文档后，AI 可以基于文档内容回答你的问题。

## 功能

- 📄 **文档上传** — 支持拖拽上传或选择文件（.txt / .pdf / .docx）
- 🔍 **智能问答** — 基于文档内容，AI 精准回答
- 🗑️ **文档管理** — 查看已加载的文件列表，随时删除
- 💾 **持久化存储** — FAISS 向量索引保存到磁盘，重启后自动恢复
- 📱 **响应式界面** — 桌面端双栏布局 + 移动端自适应

## 技术栈

| 层 | 技术 |
|------|--------|
| 后端框架 | FastAPI + Uvicorn |
| Agent 框架 | LangChain v1 (`create_agent`) |
| 大语言模型 | DeepSeek API（兼容 OpenAI 协议） |
| 向量数据库 | FAISS |
| 文本嵌入 | sentence-transformers (`all-MiniLM-L6-v2`) |
| 前端 | 纯 HTML + CSS + JS（Lucide 图标） |

## 快速开始

### 前置要求

- Python 3.10+
- DeepSeek API Key（[免费注册](https://platform.deepseek.com)）

### 1. 克隆项目

```bash
git clone https://github.com/你的用户名/kb-qa-agent.git
cd kb-qa-agent
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r backend/requirements.txt
```

### 3. 配置 API Key

```bash
cp backend/.env.example backend/.env
```

编辑 `backend/.env`，填入你的 DeepSeek API Key：

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

> **中国大陆用户**：如果 HuggingFace 下载慢，可以在 `.env` 中添加：
> ```
> HF_ENDPOINT=https://hf-mirror.com
> ```

### 4. 启动

```bash
cd backend
python app.py
```

首次启动需要 20~30 秒（LangChain Agent 初始化 + 下载嵌入模型），之后显示：

```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 5. 打开前端

浏览器打开 **http://localhost:8000**

> 后端托管了前端页面，无需额外启动 Web 服务器。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/status` | 获取已加载文件列表 |
| POST | `/api/chat` | 发送聊天消息 |
| POST | `/api/load` | 按文件路径加载文档 |
| POST | `/api/remove` | 删除已加载的文档 |
| POST | `/api/upload` | 上传文件并加载到知识库 |

## 部署到线上（让别人看到效果）

### 方式一：HuggingFace Spaces（推荐）

1. 将项目推送到 GitHub
2. 访问 [huggingface.co/spaces](https://huggingface.co/spaces) → 创建新 Space
3. 选择 **Docker** → **FastAPI**
4. 关联你的 GitHub 仓库
5. 在 Space 的 Settings → Repository Secrets 中添加：
   - `DEEPSEEK_API_KEY`：你的 API Key
   - `DEEPSEEK_BASE_URL`：API 地址
6. 创建 `Dockerfile`（见下方），Space 会自动构建并运行

#### Dockerfile（放到项目根目录）

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r backend/requirements.txt

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

> HuggingFace Spaces 默认端口为 7860，需在 `app.py` 中修改 `port=7860` 或通过环境变量配置。

### 方式二：Render.com

1. 推送代码到 GitHub
2. 在 [render.com](https://render.com) 创建 **Web Service**
3. 连接 GitHub 仓库
4. Environment 中把 DEEPSEEK_API_KEY 填进 Render 的 Environment Variables
5. 部署后 Render 会生成一个公开 URL

## 项目结构

```
kb-qa-agent/
├── backend/
│   ├── agent_core.py          # LangChain Agent 核心逻辑
│   ├── app.py                 # FastAPI 服务
│   ├── requirements.txt       # 依赖列表
│   ├── .env                   # 你的 API Key（不提交到 Git）
│   └── faiss_index/           # 向量索引（生成文件，不提交）
├── frontend/
│   └── index.html             # 前端页面
├── .gitignore
└── README.md
```

## 代码说明

- **`agent_core.py`**：定义了知识库的工具函数（加载/删除文档、检索）、Agent 创建、对话记忆。
- **`app.py`**：FastAPI 路由，提供 REST API + 前端页面托管。

## License

MIT
