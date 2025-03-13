# 📊 CSV Question Answering with AI-Powered SQL Queries

This project allows users to **upload a CSV file**, convert it into an **SQLite database**, and ask questions about the data using **natural language**. The system uses an **Ollama-based AI model** to generate SQL queries, execute them on the database, and return results in a user-friendly format. If required, the system can also generate **graph visualizations**.

## 🚀 Features

- **CSV to SQL Conversion:** Automatically converts uploaded CSV files into an SQLite database.
- **AI-Powered Queries:** Uses an Ollama model to generate SQL queries based on user input.
- **Query Execution:** Runs AI-generated SQL on the database and returns results.
- **Graph Generation:** If relevant, the AI suggests graphs, and they are dynamically generated.
- **Gradio UI:** Simple web interface for file upload, question input, and viewing results.

---

## ⚙️ How It Works

### **1️⃣ Upload a CSV File**
- The system reads the CSV file and extracts column names.
- It creates an **SQLite database** from the CSV data.
- The dataset's **summary statistics** are computed.

### **2️⃣ Initialize the AI Agent**
- The Ollama model is given:
  - The **column names** from the CSV.
  - A **statistical summary** of the data.
- The AI is prompted to **generate SQL queries** that match user queries.

### **3️⃣ Ask Questions in Natural Language**
- When the user asks a question, the system:
  1. Sends the query to the AI.
  2. The AI generates an SQL query and an optional graph suggestion.
  3. The SQL query is executed, and the result is formatted.
  4. If a graph is needed, it is generated using Matplotlib.


## 🔧 Installation & Setup

### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

## Running the project
```bash
python main.py
```
