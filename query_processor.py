import sqlite3
from config import db_name, table_name

def execute_sql_query(query: str):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        conn.close()
        return result, column_names
    except Exception:
        return None, None
