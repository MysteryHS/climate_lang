from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import render

from dotenv import load_dotenv
import os
import json

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_dir = "./db"

load_dotenv("../.env")
KEY = os.getenv("KEY")

db = Chroma(persist_directory=persist_dir, embedding_function=embedding_function)

llm_src = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=KEY)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
The answer should be easy to be understood by anyone. Always consider the question to be asked related to climate change. Your source is IPCC reports.

{context}

Question: {question}
Answer:"""

retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm_src,
    chain_type="stuff",
    retriever=db.as_retriever(
        search_kwargs={'k': 6}
    ),
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
    )},
    return_source_documents=True,
)

class Document:
    def __init__(self, text, metadata=None):
        self.text = text
        self.metadata = metadata if metadata is not None else {}

def index(request):
    return render(request, 'chatbot/index.html')


class BasicApi(APIView):
    def post(self, request):
        print(request.body)
        query = json.loads(request.body)["prompt"]
        # docs = db.similarity_search(query)
        # metadatas = []
        # for doc in docs:
        #     metadatas.append(doc.metadata)
        result = retrieval_qa({"query": query})
        # print(result)
        # result = {"query": query, "result": "oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo", "source_documents": [
        #         Document("yoo", {"page": 42, "source": "data/IPCC_AR6_WGIII_FullReport.pdf"}),
        #         Document("yoo", {"page": 43, "source": "data/IPCC_AR6_WGII_FullReport.pdf"}),
        #     ]}
        metadatas = []
        for metadata in result["source_documents"]:
            metadatas.append(metadata.metadata)
        result["source_documents"] = metadatas
        return Response({
            'result': result
        })

