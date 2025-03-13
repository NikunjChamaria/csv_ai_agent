#Formats SQL results into a readable response.
def format_response(user_prompt: str, sql_result, column_names):
    if not sql_result or not column_names:
        return "I'm sorry, but I couldn't find any answer for your question."

    formatted_data = []
    for row in sql_result:
        formatted_row = ", ".join(f"{col}: {val}" for col, val in zip(column_names, row))
        formatted_data.append(formatted_row)

    return f"Based on your question: '{user_prompt}', here is what I found:\n" + "\n".join(formatted_data)
