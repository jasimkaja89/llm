import os
import requests
import gradio as gr

# Define the Hugging Face API URL for Falcon-7B-Instruct and your API key
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"  # Falcon-7B-Instruct model URL
API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # Replace with your actual environment variable name

# Function to query the Falcon-7B-Instruct model on Hugging Face
def query_huggingface(input_text, personality):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    if personality == "Introvert":
        prompt = f"As an introspective guide, respond thoughtfully. Question: {input_text}"
    else:  # Extrovert
        prompt = f"As an energetic friend, respond lively. Question: {input_text}"

    # Make a request to the Hugging Face API
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})

    # Check if the response is successful
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return f"Error: {response.status_code}, {response.text}"

# Create the Gradio interface
def create_interface():
    iface = gr.Interface(
        fn=query_huggingface,
        inputs=[
            gr.Textbox(label="Your Question"),
            gr.Radio(choices=["Introvert", "Extrovert"], label="Choose Your Personality")
        ],
        outputs="text",
        title="Personality-Based LLM Interaction with Falcon-7B-Instruct",
        description="Select your personality type and ask a question, now using Falcon-7B-Instruct for responses."
    )
    return iface

# Launch the interface
create_interface().launch(share=True)

