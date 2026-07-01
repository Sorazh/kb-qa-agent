import aiosqlite
import os

#数据库文件路径
DB_PATH = "app_metadata.db"

async def init_db():
    """第一次运行程序时，若表不存在则创建"""
    async with aiosqlite.connect(DB_PATH) as db:
        # SQL语句：如果不存在名为 'documents' 的表，就创建它
        # id: 编号, file_path: 文件路径, chunk_count: 切成了多少块
        # created_at: 上传时间（默认存为当前UTC时间）
        # 创建的表名(documents) 列的表(id) 列的数据类型(INT) 列的约束(NOT NULL等)
        await db.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                chunk_count INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        await db.commit()  #提交执行

async def add_file_record(file_path:str,chunk_count:int):
    """记录一个文件到数据库"""
    async with aiosqlite.connect(DB_PATH) as db:
        #插入数据，如果文件重复则忽略(IGNORE)
        await db.execute(
            "INSERT OR IGNORE INTO documents (file_path, chunk_count) VALUES (?, ?)",
            (file_path, chunk_count)
        )
        await db.commit()

async def get_all_files():
    """获取所有文件路径列表（这里用于方便兼容前端）"""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute('SELECT file_path FROM documents ORDER BY created_at DESC')
        rows = await cursor.fetchall()
        #把结果转成['path1','path2']这种格式返回
        return [row[0] for row in rows]

async def delete_file_record(file_path:str):
    """根据路径删除记录"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM documents WHERE file_path = ?", (file_path,))
        await db.commit()

async def clear_all_records():
    """清空整个 documents 表（当知识库被完全清空时调用）"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM documents")  # 删除表中所有行
        await db.commit()
        print("[DB] 已清空 documents 表中的所有记录")