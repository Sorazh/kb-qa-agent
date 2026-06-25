"""
知识库问答 Agent (LangChain v1)
功能：
- 通过 load_document 工具加载 txt/pdf/docx 文件
- 通过 query_knowledge 工具检索知识库并回答
- 对话记忆（MemorySaver）
- 日志中间件
- 运行时上下文（用户 ID）
- 知识库持久化（重启后自动恢复）
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
import os
import json
from dataclasses import dataclass
from typing import Any, List
from typing_extensions import NotRequired  # TypedDict 可选字段标记
from dotenv import load_dotenv

load_dotenv()

# ---------- 1. 导入 LangChain v1 核心 ----------
from langchain_openai import ChatOpenAI
from langchain.tools import tool                     # @tool 装饰器（重新导出自 langchain-core）
from langchain.agents import create_agent, AgentState # AgentState 是 TypedDict
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# ---------- 2. 初始化模型 ----------
# 使用 DeepSeek 兼容接口（通过 OpenAI 协议调用）
model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0.7,
)

# ---------- 3. 知识库相关导入 ----------
from langchain_community.vectorstores import FAISS       # 稠密向量检索库
from langchain_huggingface import HuggingFaceEmbeddings   # sentence-transformers 嵌入
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ========== 持久化配置 ==========
FAISS_INDEX_DIR = "./faiss_index"       # FAISS 索引持久化目录
LOADED_FILES_JSON = "./loaded_files.json"  # 已加载文件记录
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 嵌入模型名称（轻量、效果好）
# ================================

# 全局知识库（惰性初始化，首次使用时从磁盘恢复或创建）
_knowledge_base: FAISS | None = None


# ============================================================
#  运行查询入口（供 FastAPI 调用，非交互式）
# ============================================================
def run_query(user_input: str, thread_id: str = "default") -> str:
    """
    给定用户输入，返回 Agent 最终文本答复。

    参数:
        user_input: 用户提问文本
        thread_id:  会话 ID（用于区分不同对话历史）

    返回:
        str: Agent 的最终回答，失败则返回 "(无响应)"
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )
    # 从结果中提取最后一个 AIMessage 的内容（即 Agent 的回答）
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            return msg.content
    return "(无响应)"


# ============================================================
#  嵌入模型与知识库辅助函数
# ============================================================

def _get_embeddings():
    """获取 HuggingFace 嵌入模型实例（本地模型，无需 API Key）"""
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


def _ensure_kb_loaded():
    """
    从磁盘恢复知识库（如果持久化索引存在）。
    重启后自动加载之前导入的文档，无需重新导入。
    """
    global _knowledge_base
    if _knowledge_base is not None:
        return
    if os.path.isdir(FAISS_INDEX_DIR) and os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")):
        _knowledge_base = FAISS.load_local(
            FAISS_INDEX_DIR,
            _get_embeddings(),
            allow_dangerous_deserialization=True,  # 本地索引安全
        )
        print(f"[KB] 已从磁盘恢复知识库：{FAISS_INDEX_DIR}")
    else:
        print("[KB] 未检测到持久化知识库，等待首次 load_document...")


def _save_kb():
    """将内存中的 FAISS 索引保存到磁盘"""
    global _knowledge_base
    if _knowledge_base is None:
        return
    _knowledge_base.save_local(FAISS_INDEX_DIR)
    print(f"[KB] 知识库已持久化到：{FAISS_INDEX_DIR}")


def _record_loaded_file(filepath: str):
    """
    记录已导入的文件路径（用于展示和去重）。
    保存在 loaded_files.json 中。
    """
    files = set()
    if os.path.exists(LOADED_FILES_JSON):
        with open(LOADED_FILES_JSON, "r", encoding="utf-8") as f:
            files = set(json.load(f))
    files.add(os.path.abspath(filepath))
    with open(LOADED_FILES_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted(files), f, ensure_ascii=False, indent=2)


def _list_loaded_files() -> list[str]:
    """返回所有已记录的文件路径列表"""
    if os.path.exists(LOADED_FILES_JSON):
        with open(LOADED_FILES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _load_file_to_documents(filepath: str) -> List[Document]:
    """
    根据文件扩展名加载文件内容，返回 Document 列表。
    支持的格式：.txt（纯文本）、.pdf（PyPDF 解析）、.docx（python-docx 解析）
    """
    ext = os.path.splitext(filepath)[1].lower()
    documents = []

    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append(Document(page_content=text, metadata={"source": filepath}))

    elif ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        documents.append(Document(page_content=text, metadata={"source": filepath}))

    elif ext == ".docx":
        from docx import Document as DocxDocument
        doc = DocxDocument(filepath)
        text = "\n".join([para.text for para in doc.paragraphs])
        documents.append(Document(page_content=text, metadata={"source": filepath}))

    else:
        raise ValueError(f"不支持的文件格式: {ext}，仅支持 .txt .pdf .docx")

    return documents


# ============================================================
#  文档加载与删除核心逻辑（do_* 函数）
#  FastAPI 直接调用这些函数，Agent 工具也调用它们
# ============================================================

def do_load_document(filepath: str) -> str:
    """
    实际的加载文档逻辑。

    流程：
        1. 确保知识库已初始化（从磁盘恢复或新建）
        2. 读取文件并切分为段落块（chunk_size=300, chunk_overlap=30）
        3. 将文本块添加到 FAISS 索引中
        4. 持久化索引到磁盘
        5. 记录已加载文件

    参数:
        filepath: 文件的绝对路径或相对路径

    返回:
        str: 操作结果描述（成功/失败）
    """
    try:
        if not os.path.exists(filepath):
            return f"文件不存在: {filepath}"

        _ensure_kb_loaded()

        docs = _load_file_to_documents(filepath)
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        chunks = splitter.split_documents(docs)

        global _knowledge_base
        embeddings = _get_embeddings()

        if _knowledge_base is None:
            _knowledge_base = FAISS.from_documents(chunks, embeddings)
        else:
            _knowledge_base.add_documents(chunks)

        _save_kb()
        _record_loaded_file(filepath)

        return (
            f"成功加载文档: {filepath}\n"
            f"本次新增片段数: {len(chunks)}\n"
            f"知识库已持久化到: {FAISS_INDEX_DIR}"
        )
    except Exception as e:
        return f"加载文档失败: {str(e)}"


def do_remove_document(filepath: str) -> str:
    """
    实际的删除文档逻辑。

    流程：
        1. 查找匹配的已加载文件
        2. 从文件列表中移除
        3. 用剩余文件重建整个知识库
        4. 更新持久化文件

    参数:
        filepath: 要删除的文件路径（自动匹配大小写和格式）

    返回:
        str: 操作结果描述（成功/失败）
    """
    try:
        abs_path = os.path.abspath(filepath)
        current_files = _list_loaded_files()
        if not current_files:
            return "当前知识库中没有记录任何文件，无法删除。"

        # 不区分大小写匹配文件路径
        matched = None
        for f in current_files:
            if os.path.normcase(f) == os.path.normcase(abs_path):
                matched = f
                break

        if matched is None:
            return (f"未找到已导入的文件：{filepath}\n"
                    f"当前已导入的文件：\n" + "\n".join(current_files))

        new_files = [f for f in current_files if f != matched]
        rebuild_knowledge_base_from_files(new_files)

        with open(LOADED_FILES_JSON, "w", encoding="utf-8") as f:
            json.dump(sorted(new_files), f, ensure_ascii=False, indent=2)

        return (f"✅ 已成功删除文档：{matched}\n"
                f"知识库已重建并持久化。当前共 {len(new_files)} 个文件。")
    except Exception as e:
        return f"删除文档失败: {str(e)}"

def rebuild_knowledge_base_from_files(filepaths: List[str]):
    """
    根据给定的文件列表重新构建整个知识库（清空旧索引）。

    用于删除文档后的重建：将剩余文件全部重新切分、向量化，
    创建全新的 FAISS 索引。

    参数:
        filepaths: 保留的文件路径列表（这些文件将被重新索引）
    """
    global _knowledge_base
    _knowledge_base = None

    if not filepaths:
        import shutil
        if os.path.isdir(FAISS_INDEX_DIR):
            shutil.rmtree(FAISS_INDEX_DIR)
        if os.path.exists(LOADED_FILES_JSON):
            os.remove(LOADED_FILES_JSON)
        return

    all_chunks = []
    embeddings = _get_embeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

    for fp in filepaths:
        if not os.path.exists(fp):
            continue
        docs = _load_file_to_documents(fp)
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    if all_chunks:
        _knowledge_base = FAISS.from_documents(all_chunks, embeddings)
        _save_kb()
    else:
        _knowledge_base = None


# ============================================================
#  4. Agent 工具定义
# ============================================================

@tool
def load_document(filepath: str) -> str:
    """
    加载一个文档文件（txt/pdf/docx）到知识库中，供后续查询使用。
    文件路径必须是绝对路径或相对于当前工作目录的有效路径。
    加载后自动持久化，下次启动无需重新导入。
    """
    # 直接调用核心函数 do_load_document，避免重复执行
    return do_load_document(filepath)

@tool
def remove_document(filepath: str) -> str:
    """
    从知识库中删除一个已导入的文档（支持 .txt/.pdf/.docx）。
    输入可以是绝对路径或相对路径，程序会自动匹配已记录的文件。
    删除后知识库会自动重建并持久化。
    """
    # 直接调用核心函数 do_remove_document，避免重复执行
    return do_remove_document(filepath)

@tool
def query_knowledge(query: str) -> str:
    """
    从已加载的知识库中检索与查询最相关的信息并返回。
    请先使用 load_document 加载至少一个文档后再调用此工具。
    """
    _ensure_kb_loaded()

    global _knowledge_base
    if _knowledge_base is None:
        loaded = _list_loaded_files()
        hint = f"\n（已记录的导入文件：{loaded}）" if loaded else ""
        return "知识库为空，请先使用 load_document 工具加载文档。" + hint

    results = _knowledge_base.similarity_search(query, k=3)
    if not results:
        return "未找到相关信息。"

    combined = "\n---\n".join([doc.page_content for doc in results])
    return f"找到以下相关内容：\n{combined}"

# ============================================================
#  5. 自定义状态和上下文
# ============================================================

class MyAgentState(AgentState):
    """
    自定义 Agent 状态，在基础消息列表之外增加用户名字段。

    注意：AgentState 是 TypedDict（Python 3.8+），
    可选字段必须用 NotRequired 标记，不能直接赋默认值。
    """
    user_name: NotRequired[str | None]


# ============================================================
#  6. 中间件（可选） — 基于 AgentMiddleware
# ============================================================

class LoggingMiddleware(AgentMiddleware):
    """
    日志中间件：在模型调用前后打印调试信息。
    before_model: 模型调用前触发
    after_model:  模型调用后触发
    """

    def before_model(self, state: MyAgentState, runtime) -> dict[str, Any] | None:
        print(f"[中间件] 模型调用前，消息数：{len(state['messages'])}")
        return None

    def after_model(self, state: MyAgentState, runtime) -> dict[str, Any] | None:
        last = state["messages"][-1] if state["messages"] else None
        if isinstance(last, AIMessage):
            print(f"[中间件] 模型响应长度：{len(last.content)} 字符")
        return None


# ============================================================
#  7. 创建 Agent
# ============================================================

agent = create_agent(
    model=model,
    tools=[load_document, query_knowledge, remove_document],
    system_prompt=(
        "你是一个知识库问答助手。你的工作流程如下：\n\n"
        "1. 当用户提供文件路径（以.txt、.pdf 或.docx 结尾）时，"
        "立即调用 load_document 工具加载该文件。\n"
        "2. 当用户提出问题时，**必须首先调用 query_knowledge 工具**尝试检索知识库。\n"
        "   - 如果检索结果不为空，则基于检索结果回答。\n"
        "   - 如果检索结果为空（工具返回\u201c知识库为空\u201d或\u201c未找到相关信息\u201d），"
        "再询问用户是否需要加载文档。\n"
        "3. 知识库支持磁盘持久化，重启后会自动恢复，"
        "因此即使你没有看到用户手动加载过文档，也应该先尝试 query_knowledge。\n"
        "4. 当用户要求删除某个已导入的文档时，调用 remove_document 工具，"
        "并提供正确的文件路径。\n"
        "5. 不要在未调用工具的情况下凭空回答知识库相关的问题。\n\n"
        "可用工具：\n"
        "- load_document(filepath): 加载文档到知识库（自动持久化）\n"
        "- query_knowledge(query): 检索知识库并返回相关内容\n"
        "- remove_document(filepath): 从知识库中删除指定文档（自动重建索引）"
    ),
    state_schema=MyAgentState,
    middleware=[LoggingMiddleware()],
    checkpointer=MemorySaver(),
)

# ============================================================
#  8. 主交互循环（控制台使用）
# ============================================================

def main():
    """命令行交互入口，用于本地测试 Agent 功能。"""
    print("=== 知识库问答 Agent (LangChain v1) ===")
    print("支持文件格式：.txt .pdf .docx")
    print("知识库持久化目录:", FAISS_INDEX_DIR)
    print("已导入文件记录:", LOADED_FILES_JSON)

    # 启动时显示已有的导入记录
    existing_files = _list_loaded_files()
    if existing_files:
        print("上次已导入的文件：")
        for f in existing_files:
            print(f"  - {f}")

    thread_id = input("请输入会话ID（留空自动生成）: ") or "default-thread"
    user_id = input("请输入你的用户名: ") or "anonymous"

    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ("exit", "quit", "拜拜"):
            print("Agent: 再见！")
            break

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                print(f"Agent: {msg.content}")
                break


if __name__ == "__main__":
    main()
