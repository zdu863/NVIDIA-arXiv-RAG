import os
import random
import glob
import json
import time

from copy import deepcopy
from datetime import datetime
from fastapi import FastAPI

from operator import itemgetter
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough, chain

from langchain_core.runnables.passthrough import RunnableAssign
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import ArxivLoader

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import ChatPromptValue
from functools import partial
from operator import itemgetter

from langserve import RemoteRunnable
import gradio as gr
from typing import List

import logging
import traceback
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore

# Access the environment variable
nvidia_api_key = os.environ.get('NVIDIA_API_KEY')

embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END", api_key = nvidia_api_key)

instruct_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", api_key = nvidia_api_key)

embed_dims = len(embedder.embed_query("test"))
def default_FAISS():
    '''Useful utility for making an empty FAISS vectorstore'''
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

all_docstores = {name: FAISS.load_local(name, embedder, allow_dangerous_deserialization=True) for name in glob.glob('*_docstore_index')}

all_retrievers = lambda x: all_docstores[x].as_retriever()

def get_traceback(e):
    lines = traceback.format_exception(type(e), e, e.__traceback__)
    return ''.join(lines)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#####################################################################
## Chain Dictionary

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        if isinstance(doc, dict):
            out_str += doc.get('page_content', doc) + "\n"
        else: 
            out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str


def output_puller(inputs):
    """If you want to support streaming, implement final step as a generator extractor."""
    docs = ''
    for token in inputs:
        if token.get('output'):
            yield token.get('output')
        if token.get('context_raw'):
            docs = token.get('context_raw')
    yield "\n\n References: \n" 
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', "Document")
        if doc_name:
            yield f"[Quote from {doc_name}] "
            yield "\n"
        if isinstance(doc, dict):
            yield doc.get('page_content', doc)
            yield "\n"
        else: 
            yield getattr(doc, 'page_content', str(doc)) 
            yield "\n"

        
def RPrint(preface=""):
    """Simple passthrough "prints, then returns" chain"""
    def print_and_return(x, preface):
        if preface: print(preface, end="")
        print(x)
        print(instruct_llm)
        # print(doccstore)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

#####################################################################
## Chain Structure

## Basic Chat Chain

basic_chat_prompt = ChatPromptTemplate.from_messages([("system",
    "You are a chatbot. Help the user as they ask questions. "
    "You have some memory of the chat history that you can use for context:\n{history}\n\n"
), ('user', '{input}')])

basic_chain = (
    {'input' : (lambda x: x['input']),
     'history': (lambda x: ''.join("[User]: " + it[0] +"\n"+ "[Chatbot]: " + it[1] + "\n" for it in x['history'][-2:-20:-1]) if len(x['history'])>1 else '')}
    | basic_chat_prompt 
    | instruct_llm
    # | RPrint()
)
    
## Retrieval-Augmented Generation Chain

def assert_docs(d):
    if isinstance(d, list) and len(d) and isinstance(d[0], (Document, dict)):
        return d
    gr.Warning(f"Retriever outputs should be a list of documents, but instead got {str(d)[:100]}...")
    return []

## Conversation Retrival


convstore = default_FAISS()

def save_memory_and_get_output(d, vstore):
    """Accepts 'input'/'output' dictionary and saves to convstore"""
    vstore.add_texts([
        f"User previously responded with {d.get('input')}",
        f"Agent previously responded with {d.get('output')}"
    ])
    return d.get('output')

@chain
def custom_chain(input_dict):
    retriever = all_retrievers(input_dict['doc_index'])
    return retriever.invoke(input_dict['input'])

chat_prompt = ChatPromptTemplate.from_messages([("system",
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked: {input}\n\n"
    " From this, we have retrieved the following potentially-useful info: "
    " Conversation History Retrieval:\n{rag_history}\n\n"
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational.)"
), ('user', '{input}')])

## Document Retrival

retrieval_chain = (
    {'input' : (lambda x: x['input']), 'doc_index': (lambda x: x['doc_index'])}
    # | RunnableAssign(
    #     {'retriever': itemgetter('doc_index') | all_docstores_func }
    # )
    | RunnableAssign(
        {'context_raw' : custom_chain
        | assert_docs
        | LongContextReorder().transform_documents
    })
    | RunnableAssign(
        {'context' : RunnableLambda(itemgetter('context_raw'))
        | docs2str
    })
    | RunnableAssign(
        {'rag_history' : itemgetter('input') 
        | convstore.as_retriever() 
        | LongContextReorder().transform_documents | docs2str})
    | RPrint()
)

output_chain = RunnableAssign({"output" : chat_prompt | instruct_llm |  StrOutputParser()}) | output_puller
rag_chain = retrieval_chain | output_chain

#####################################################################
## Create new document index

def new_docstore(arxiv_list, vname, progress = gr.Progress()):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100,
        separators=["\n\n", "\n", ".", ";", ",", " "],
    )
    progress(0, desc="Starting")
    
    try:
        arxiv_list = arxiv_list.replace(" ", "").split(",")
        docs = []
        for ii, ind in enumerate(arxiv_list):
            docs.append(ArxivLoader(query=ind).load())
            progress(ii*(0.3)/len(docs), desc="Getting documents")
            
    except Exception as e:
        logger.error(f"Loading arXiv articles failed:\n{get_traceback(e)}")
        logger.error(f"Please check you entered the correct arXiv identifiers.")
    
    ## Cut the paper short if references is included.
    ## This is a standard string in papers.
    for ii, doc in enumerate(docs):
        content = json.dumps(doc[0].page_content)
        logger.error(f"{content}")
        if "References" in content:
            doc[0].page_content = content[:content.index("References")]
        progress(0.3+ii*(0.6-0.3)/len(docs), desc="Chunking documents")

    ## Split the documents and also filter out stubs (overly short chunks)

    docs_chunks = [text_splitter.split_documents(doc) for doc in docs]
    docs_chunks = [[c for c in dchunks if len(c.page_content) > 200] for dchunks in docs_chunks]
    
    ## Make some custom Chunks to give big-picture details
    
    doc_string = "Available Documents:"
    doc_metadata = []
    for chunks in docs_chunks:
        metadata = getattr(chunks[0], 'metadata', {})
        doc_string += "\n - " + metadata.get('Title')
        doc_metadata += [str(metadata)]
    
    extra_chunks = [doc_string] + doc_metadata
    vecstores = [FAISS.from_texts(extra_chunks, embedder)]
    
    for ii,doc_chunk in enumerate(docs_chunks):
        vecstores.append(FAISS.from_documents(doc_chunk, embedder))
        progress(0.6+ii*(0.99-0.6)/len(docs_chunks), desc="Creating vecstores")

    progress(0.98, desc="Merging vecstores")
    
    ## Initialize an empty FAISS Index and merge others into it
    ## We'll use default_faiss for simplicity, though it's tied to your embedder by reference
    
    agg_vstore = default_FAISS()
    for vstore in vecstores:
        agg_vstore.merge_from(vstore)
        
    # if f"{vname}_docstore_index" in glob.glob('*_docstore_index'):
    #     logger.error(f"Loading arXiv articles failed:\n{get_traceback(e)}")

    agg_vstore.save_local(f"{vname}_docstore_index")
    time.sleep(5)
    
    return "Index created!"


#####################################################################
## ChatBot utilities

def add_message(message, history, role=0, preface=""):
    if not history or history[-1][role] is not None:
        history += [[None, None]]
    history[-1][role] = preface
    buffer = ""
    try:
        for chunk in message:
            token = getattr(chunk, 'content', chunk)
            buffer += token
            history[-1][role] += token
            yield history, buffer, False 
            
        save_memory_and_get_output({'input':  message, 'output': buffer}, convstore)

    except Exception as e:
        logger.error(f"Gradio Stream failed:\n{get_traceback(e)}")
        history[-1][role] += f"...\nGradio Stream failed: {e}"
        yield history, buffer, True


def add_text(history, text, llm_model, doc_index):
    global instruct_llm
    instruct_llm = ChatNVIDIA(model=llm_model, api_key = nvidia_api_key)
    # doccstore = all_docstores[doc_index]
    
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def bot(history, chain_key, doc_ind):
    chain = {'Basic' : basic_chain, 'RAG' : rag_chain}.get(chain_key)
    msg_stream = chain.stream({'input':history[-1][0], 'history':history, 'doc_index':doc_ind})
    for history, buffer, is_error in add_message(msg_stream, history, role=1):
        yield history


#####################################################################
## GRADIO EVENT LOOP

# https://github.com/gradio-app/gradio/issues/4001
CSS ="""
.contain { display: flex; flex-direction: column; height:120vh;}
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""
THEME = gr.themes.Default(primary_hue="green")

def update_dropdown_options(text_input):
    # Example: Create dropdown options based on words in the textbox
    options = text_input.split()  # Split text into words as options
    return gr.update(choices=glob.glob('*_docstore_index'))  # Update dropdown with new options

def update_docstore(text_input):
    # Example: Create dropdown options based on words in the textbox
    global all_docstores
    if text_input not in all_docstores:
        all_docstores[text_input] = FAISS.load_local(text_input, embedder, allow_dangerous_deserialization=True)
        
def get_demo():
    with gr.Blocks(css=CSS, theme=THEME) as demo:
        with gr.Column():
            arxiv_ids = gr.Textbox(
                scale=4,
                label = "Create a new document index and name it (This is not necessary for chatting if you want to use an existing document index).",
                show_label=True,
                placeholder="Enter a list of arXiv identifiers, separated by commas. For example, 1706.03762, 1810.04805, 2005.11401, 2205.00445"
                # container=False,
            )
            with gr.Row():
                docstore_name = gr.Textbox(
                    scale=4,
                    show_label=False,
                    placeholder="Enter a name for the document index (only alphanumberical and underscore)."
                    # container=False,
                )
                progress_output = gr.Textbox(show_label=False, placeholder="Progress")
                create_btn   = gr.Button("Create new document index")
        
        test_msg = create_btn.click(
            new_docstore, 
            inputs=[arxiv_ids, docstore_name],
            outputs = progress_output
        )
            
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            bubble_full_width=False,
            avatar_images=(None, (os.path.join(os.path.dirname(__file__), "parrot.png"))),
        )

        with gr.Row(): # llm model, embedding model and index selection
            model_choices = sorted([model.id for model in ChatNVIDIA.get_available_models(api_key = nvidia_api_key)])
            llm_model = gr.Dropdown(choices=model_choices, value = "meta/llama-3.1-8b-instruct", label="Select LLM Model")

            available_index = glob.glob('*_docstore_index')
            doc_index = gr.Dropdown(choices=available_index, value = "default_docstore_index", label="Select Document Index")

        progress_output.change(fn=update_dropdown_options, inputs=progress_output, outputs=doc_index)
        
        doc_index.input(fn=update_docstore, inputs=doc_index, outputs=[])


        with gr.Row():
            txt = gr.Textbox(
                scale=4,
                show_label=False,
                placeholder="Enter text and press enter.",
                container=False,
            )

            chain_btn  = gr.Radio(["Basic", "RAG"], value="Basic", label="Main Route")
            # test_btn   = gr.Button("🎓\nEvaluate")

        
        # Reference: https://www.gradio.app/guides/blocks-and-event-listeners

        # This listener is triggered when the user presses the Enter key while the Textbox is focused.
        txt_msg = (
            # first update the chatbot with the user message immediately. Also, disable the textbox
            txt.submit(              ## On textbox submit (or enter)...
                fn=add_text,            ## Run the add_text function...
                inputs=[chatbot, txt, llm_model, doc_index],  ## Pass in the values of chatbot and txt...
                outputs=[chatbot, txt], ## Assign the results to the values of chatbot and txt...
                queue=False             ## And don't use the function as a generator (so no streaming)!
            )
            # then update the chatbot with the bot response (same variable logic)
            .then(bot, [chatbot, chain_btn, doc_index], [chatbot])
            ## Then, unblock the textbox by assigning an active status to it
            .then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
        )

        # test_msg = test_btn.click(
        #     rag_eval, 
        #     inputs=[chatbot, chain_btn], 
        #     outputs=chatbot, 
        # )

    return demo

#####################################################################
## Final App Deployment

if __name__ == "__main__":

    demo = get_demo()
    demo.queue()

    # logger.warning("Starting FastAPI app")
    app = FastAPI()

    app = gr.mount_gradio_app(app, demo, '/')

    @app.route("/health")
    async def health():
        return {"success": True}, 200
