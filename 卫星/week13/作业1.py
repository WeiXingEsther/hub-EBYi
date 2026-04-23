import re
import sqlite3
from typing import Any, Dict, List

from openai import OpenAI


class DBParser:
    """解析 SQLite 数据库结构"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def get_table_names(self) -> List[str]:
        """
        获取所有用户表
        排除 SQLite 系统表
        """
        sql = """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name;
        """
        rows = self.conn.execute(sql).fetchall()
        return [row["name"] for row in rows]

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """获取某张表的字段信息"""
        rows = self.conn.execute(f"PRAGMA table_info({table_name});").fetchall()
        schema = []
        for row in rows:
            schema.append({
                "cid": row["cid"],
                "name": row["name"],
                "type": row["type"],
                "notnull": row["notnull"],
                "default_value": row["dflt_value"],
                "pk": row["pk"]
            })
        return schema

    def get_schema_text(self) -> str:
        """将数据库 schema 格式化成 prompt 可用文本"""
        tables = self.get_table_names()
        schema_text_list = []

        for table in tables:
            fields = self.get_table_schema(table)
            field_text = ", ".join([f"{f['name']} ({f['type']})" for f in fields])
            schema_text_list.append(f"表 {table}: {field_text}")

        return "\n".join(schema_text_list)

    def execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        """执行 SQL 并返回结果"""
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return [dict(row) for row in rows]

    def close(self):
        self.conn.close()


class NL2SQLAgent:
    """自然语言转 SQL 问答 Agent"""

    def __init__(
        self,
        db_path: str,
        api_key: str,
        base_url: str,
        model: str = "qwen2.5-7b-instruct"
    ):
        self.db = DBParser(db_path)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    def build_prompt(self, question: str) -> str:
        schema_text = self.db.get_schema_text()

        prompt = f"""
你是一个专业的 NL2SQL 助手。
你的任务是根据用户问题，基于给定的 SQLite 数据库结构，生成一条正确的 SQL 查询语句。

要求：
1. 只输出 SQL，不要输出解释，不要输出 markdown。
2. 必须基于给定 schema 生成 SQL。
3. 如果是统计数量，优先使用 COUNT(*)。
4. 如果涉及多个表的数量统计，可以使用子查询并分别命名字段。
5. 表名和字段名必须与 schema 完全一致。
6. 数据库类型是 SQLite。

数据库 schema 如下：
{schema_text}

用户问题：
{question}
"""
        return prompt.strip()

    def call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个严谨的 SQLite NL2SQL 助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def clean_sql(self, sql_text: str) -> str:
        sql_text = sql_text.strip()

        sql_text = re.sub(r"^```sql\s*", "", sql_text, flags=re.IGNORECASE)
        sql_text = re.sub(r"^```\s*", "", sql_text)
        sql_text = re.sub(r"\s*```$", "", sql_text)

        sql_text = sql_text.strip().split(";")[0].strip() + ";"
        return sql_text

    def answer_rewrite(self, question: str, sql: str, result: List[Dict[str, Any]]) -> str:
        if not result:
            return f"问题：{question}\nSQL：{sql}\n结果为空。"

        if len(result) == 1 and len(result[0]) == 1:
            value = list(result[0].values())[0]
            return f"{question} 答案是：{value}。"

        # 多字段结果
        parts = []
        for k, v in result[0].items():
            parts.append(f"{k} 为 {v}")
        return f"{question} 答案是：{'，'.join(parts)}。"

    def ask(self, question: str) -> Dict[str, Any]:
        prompt = self.build_prompt(question)
        raw_sql = self.call_llm(prompt)
        sql = self.clean_sql(raw_sql)

        result = self.db.execute_sql(sql)
        final_answer = self.answer_rewrite(question, sql, result)

        return {
            "question": question,
            "sql": sql,
            "result": result,
            "answer": final_answer
        }

    def close(self):
        self.db.close()


if __name__ == "__main__":
    API_KEY = "sk-2583d9d000d642e98254164d7aeb532d"
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL = "qwen2.5-7b-instruct"

    agent = NL2SQLAgent(
        db_path="chinook.db",
        api_key=API_KEY,
        base_url=BASE_URL,
        model=MODEL
    )

    questions = [
        "数据库中总共有多少张表？",
        "员工表中有多少条记录？",
        "在数据库中所有客户个数和员工个数分别是多少？"
    ]

    for q in questions:
        print("=" * 80)
        response = agent.ask(q)
        print("问题：", response["question"])
        print("生成SQL：", response["sql"])
        print("执行结果：", response["result"])
        print("最终回答：", response["answer"])

    agent.close()
