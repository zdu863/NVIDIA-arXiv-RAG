### **RAG Agents with LLMs for Context Retrieval from arXiv articles**

A web app allowing users to create vector stores from a provided list of arXiv articles. LLMs can answer queries based only on information from the vector store. \#NVIDIADevContest \#LlamaIndex


To Run the web app locally:
You need to create an account at [NVIDIA explore](https://build.nvidia.com/explore/discover) and obtain your NVIDIA api key, then do
```
export NVIDIA_API_KEY="your-api-key"
pip install -r requirements.txt
cd frontend
gradio frontend_server.py 
```