import sqlite3
import pandas as pd
from config import db_name, table_name


#Uploads CSV data into an SQLite database.
def create_sqlite_db_from_csv(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return "The uploaded CSV file is empty."
        
        conn = sqlite3.connect(db_name)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        return df
    except pd.errors.ParserError:
        return "Error parsing the CSV file. Ensure it has a valid format."
    except Exception as e:
        return f"Unexpected error: {e}"
