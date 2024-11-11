import uvicorn
from fastapi import FastAPI
import gradio as gr

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#####################################################################
## Final App Deployment

from frontend_block import get_demo

## If you do not have access to :8090, feel free to use /8090

demo = get_demo()
demo.queue()

logger.warning("Starting FastAPI app")
app = FastAPI()

app = gr.mount_gradio_app(app, demo, '/')

@app.route("/health")
async def health():
    return {"success": True}, 200

if __name__ == "__main__":
    uvicorn.run("frontend_server:app", host="0.0.0.0", port=8000, reload=False, log_level="debug")