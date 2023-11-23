import os
from dotenv import load_dotenv
load_dotenv()

from langchain.llms import GooglePalm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PDFPlumberLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


raj_llm = GooglePalm(google_api_key=os.environ['palm_api_key'], temperature=0.7)
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

vectordb_file_path = 'faiss_index'

def create_vector_db():
    # Load data from pdf/docs
    loader = PDFPlumberLoader("CNN.pdf")
    docs = loader.load()
    # Load data from pdf/

    # Create FAISS instance for  vector database from docs
    vectordb = FAISS.from_documents(documents=docs, embedding=instructor_embeddings)

    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

# create a retriever for querying the vector database
    retriever = vectordb.as_retriever()

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=['context', 'question']
    )

    # Create the RetrievalQA chain
    chain = RetrievalQA.from_chain_type(llm=raj_llm,
                                     chain_type="stuff",
                                     retriever=retriever,
                                     input_key='query',
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': PROMPT})

    return chain

if __name__ == '__main__':
    create_vector_db()
    chain = get_qa_chain()

    print(chain('CNN key terms?'))



