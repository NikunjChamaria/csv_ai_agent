#Checker file to check which ollama model has the least speed

import time
import ollama
import pandas as pd

models = ["llama3.1:8b-instruct-q2_K","llama3.1:8b-instruct-q4_K_M","llama3.1:8b","llama3.1:latest "]

prompt = "What is the capital of France?"

results = {}

for model in models:
    start_time = time.time()
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    end_time = time.time()

    response_time = end_time - start_time
    results[model] = response_time

    print(f"Model: {model}")
    print(f"Response Time: {response_time:.3f} seconds")
    print(f"Response: {response['message']['content'][:100]}...\n")

fastest_model = min(results, key=results.get)
print(f"Fastest Model: {fastest_model} ({results[fastest_model]:.3f} seconds)")