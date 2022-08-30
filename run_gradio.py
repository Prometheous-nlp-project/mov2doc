import gradio as gr
from mov2doc.pipeline import pipeline

if __name__ == "__main__":
    with gr.Blocks() as demo:
        url = gr.Textbox(label="Insert Youtube URL or Text")
        func_type = gr.Radio(["url", "text"], label="Your Input Type")
        query = gr.Textbox(label="Query")
        text = gr.Textbox(label="Text")
        keywords = gr.Textbox(label="Keywords")
        summarization = gr.Textbox(label="Summarization")
        result = gr.Textbox(label="Content Related to Query")
        qa = gr.Textbox(label="Answer to Query")
        msg = gr.Textbox(label="Message From Server")
        greet_btn = gr.Button("Summarize")
        greet_btn.click(fn=pipeline, inputs=[url, func_type, query], outputs=[
                        text, keywords, summarization, result, qa, msg])

    demo.launch()
