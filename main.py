import gradio as gr
from file_handler import create_sqlite_db_from_csv
from llm_agent import initialize_agent, ask_ai
from query_processor import execute_sql_query
from graph_plotter import generate_graph
from utils import format_response

agent = None 
df = None 

#Handles CSV file upload and AI system initialization.
def upload_csv_and_initialize(csv_path):
    global agent
    global df
    df = create_sqlite_db_from_csv(csv_path)   
    columns = ", ".join(df.columns)
    describe_data = df.describe().to_string()
    agent = initialize_agent(columns, describe_data)
    
    return "CSV uploaded & AI system updated!"

#Processes the user's question and returns AI + SQL response.
def process_query(user_prompt: str):
    global agent
    global df
    if agent is None:
        return "Please upload a CSV file first.", None

    response_data = ask_ai(agent, user_prompt)
    if "error" in response_data:
        return response_data["error"], None

    final_response = ""
    graph_image = None

    if response_data.query:
        sql_result, column_names = execute_sql_query(response_data.query)
        if sql_result:
            final_response +="\n" + format_response(user_prompt, sql_result, column_names)

    if response_data.text:
        final_response +=  response_data.text

    if response_data.graph:
        graph_image = generate_graph(response_data.graph,df=df)

    return final_response, graph_image

with gr.Blocks() as demo:
    gr.Markdown("### 📊 CSV Data Q&A with Graphs")
    csv_file = gr.File(label="Upload CSV")
    upload_status = gr.Textbox(label="Status", interactive=False)
    user_prompt = gr.Textbox(label="Ask a question")
    submit_btn = gr.Button("Get Answer")
    response_box = gr.Textbox(label="Response", interactive=False)
    graph_output = gr.Plot(label="Generated Graph")

    csv_file.upload(upload_csv_and_initialize, inputs=[csv_file], outputs=[upload_status])
    submit_btn.click(process_query, inputs=[user_prompt], outputs=[response_box, graph_output])

demo.launch()
