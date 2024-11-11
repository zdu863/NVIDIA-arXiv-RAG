# import gradio as gr

# def greet(name):
#     return "Hello " + name + "!"

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
# demo.launch()   

import uvicorn
from fastapi import FastAPI
import gradio as gr
app = FastAPI()
@app.get("/")
def read_main():
    return {"message": "This is your main app"}
io = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")
app = gr.mount_gradio_app(app, io, path="/gradio")

if __name__ == "__main__":
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=False, log_level="debug")
