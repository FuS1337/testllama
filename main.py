import constants
import time
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
from huggingface_hub import hf_hub_download
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain import PromptTemplate
from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


start_time_ges = time.time()
#web_links = constants.links_ohneanmelden
web_links = ["https://learning.sap.com/learning-journeys/acquire-core-abap-skills/preparing-the-development-environment_bc84941b-b4e6-4a6a-9b71-bb5b80e4a4ce", "https://learning.sap.com/learning-journeys/acquire-core-abap-skills/taking-a-first-look-at-abap_deb518e7-5030-48b9-9389-6507b48cf524"]
DATA_PATH = 'data/run1/'
DB_FAISS_PATH = 'vectorstore/db_faiss/run1/'
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# --------------------------------Hilfsfunktionen---------------------------------------------
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

def load_llm():
    model_path = hf_hub_download(repo_id="TheBloke/Llama-2-7B-Chat-GGML", filename="llama-2-7b-chat.ggmlv3.q8_0.bin")
    llm = CTransformers(
        model=model_path,
        model_type="llama",
        max_new_tokens=2048,
        temperature=0.01
    )
    return llm

def qa_bot(usefullllama=True):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    model = "meta-llama/Llama-2-7b-chat-hf"
    llama_pipeline = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    llm = HuggingFacePipeline(pipeline = llama_pipeline)
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa


def create_vector_db(links=web_links):
    loader = WebBaseLoader(links)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=80)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)




# --------------------------------Vectorstore aufbauen----------------------------------------
create_vector_db()
# --------------------------------Chain laden-------------------------------------------------
chain = qa_bot()
tickets = ["How do I perform a Bitlocer recovery?"]

# --------------------------------Erbenisse in Datei schreiben--------------------------------
end_time_ges = time.time()
f = open("Llama-2-7b-chat-hf_result_run1.txt", "a")
f.write("LLaMa 2 7b chat Lauf 1\n")
f.write("Startzeit: ")
f.write(str(end_time_ges - start_time_ges))
f.write("seconds\n")
f.write("#####################################################################################################")
f.write("\n")
f.write("\n")
for i in range(len(tickets)):
    print("#####")
    print(i)
    start_time = time.time()
    answer = chain.invoke({"query": tickets[i]})
    end_time = time.time()
    f.write("Ticket: ")
    f.write(tickets[i])
    f.write("\n")
    f.write("Antwort: ")
    f.write(answer["result"])
    f.write("\n")
    f.write("Zeit in Sekunden: ")
    f.write(str(end_time - start_time))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------------------------\n")
f.close()
