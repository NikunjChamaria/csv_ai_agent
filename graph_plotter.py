import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
from config import db_name, table_name
from llm_agent import Graph

#Generates a graph based on AI instructions.
def generate_graph(graph_data:Graph):
    try:
        if not graph_data or "type" not in graph_data:
            return None

        conn = sqlite3.connect(db_name)

        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()

        if not columns:
            return None  
        
        x_column = graph_data.x or columns[0]  
        y_column = graph_data.y or (columns[1] if len(columns) > 1 else columns[0])  
        conn = sqlite3.connect(db_name)
        df = pd.read_sql_query(f"SELECT {x_column}, {y_column} FROM {table_name}", conn)
        conn.close()

        fig, ax = plt.subplots(figsize=(6, 4))

        if "bar" in graph_data["type"]:
            ax.bar(df[x_column], df[y_column])
        elif "line" in graph_data["type"]:
            ax.plot(df[x_column], df[y_column], marker="o")
        elif "scatter" in graph_data["type"]:
            ax.scatter(df[x_column], df[y_column])
        else:
            return None

        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(f"{graph_data['type'].capitalize()} Chart of {x_column} vs {y_column}")

        return fig 
    except Exception as e:
        print(f"Graph Error: {e}")
        return None
