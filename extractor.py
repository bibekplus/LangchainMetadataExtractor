from langchain.prompts import PromptTemplate
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
import pandas
import gradio as gr
import json
from datetime import date
from dotenv import load_dotenv

load_dotenv()


class ScienceDirectDocument(BaseModel):
    journal: str
    paper_title:str
    authors: List[str]
    abstract: str
    publication_date: date
    keywords: List[str]
    doi: str
    topics: List[str] = Field(default_factory=list, description="Topics covered by the document. Examples include Computer Science, Biology, Chemistry, etc.")



def process_and_classify_document(filename):
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()

    ids = [str(i) for i in range(1, len(pages) + 1)]

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(pages, embeddings, ids=ids)
    retriever = vectordb.as_retriever()

    schema = ScienceDirectDocument.model_json_schema()

    template = """Use the following pieces of context to accurately classify the documents based on the schema passed. Output should follow the pattern defined in schema.
    No verbose should be present. Output should follow the pattern defined in schema and the output should be in json format only so that it can be directly used with json.loads():
    {context}
    schema: {schema}
    """
    rag_prompt_custom = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
        {"context": retriever | format_docs, "schema": RunnablePassthrough()}
        | rag_prompt_custom
        | llm
        | StrOutputParser()
    )
    output = json.loads(rag_chain.invoke(str(schema)))
    vectordb._collection.delete(ids=[ids[-1]])
    return output
    


def send_request(fileobj):
    json_output = process_and_classify_document(filename=fileobj)

    return json_output

demo = gr.Interface(send_request, 
                    gr.File(), 
                    gr.JSON(label="JSON Output"),
                    title="Document Classifier",
                    description='''A tool to classify articles from ScienceDirect. 
                    \n Disclaimer: It is a prototype and output may not be always accurate.
                    \n It is the responsibility of the user to verify the authenticity of the answer.
                    ''') 


if __name__ == "__main__":
    demo.launch()

