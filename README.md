### **RAG Agents with LLMs for Context Retrieval from arXiv articles**

A web app allowing users to create vector stores from a provided list of arXiv articles. LLMs can answer queries based only on information from the vector store.

To Run the web server:
```
pip install -r requirements.txt
cd frontend
gradio frontend_server.py 
```