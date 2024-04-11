from trulens_eval import TruChain, Tru
from trulens_eval.feedback.provider import OpenAI
from trulens_eval import Feedback
from trulens_eval.app import App
from trulens_eval.feedback import Groundedness
import numpy as np
from trulens_eval.app import App
tru = Tru()
import streamlit as sl
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

#Functions for RAG
def load_knowledgeBase():
        embeddings=OpenAIEmbeddings(model='text-embedding-3-small')
        DB_FAISS_PATH = 'faiss_products_recursive'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization = True)
        return db

def load_llm():
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        return llm

def load_prompt():
        prompt = """
        Use following piece of context to answer the question. 
        If you don't know the answer, just say you don't know. 

        Context: {context}
        Question: {question}
        Answer: 

        """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

knowledgeBase=load_knowledgeBase()
llm=load_llm()
prompt=load_prompt()

retriever = knowledgeBase.as_retriever()

rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )

#Trulens implementation
provider = OpenAI("gpt-3.5-turbo")

# select context to be used in feedback. the location of context is app specific.
context = App.select_context(rag_chain)

from trulens_eval.feedback import Groundedness
grounded = Groundedness(groundedness_provider=provider)
# Define a groundedness feedback function
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name = "Groundedness")
    .on(context.collect()) # collect context chunks into a list
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance, name = "Answer Relevance")
    .on_input_output()
)
# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name = "Context Relevance")
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

tru_recorder = TruChain(rag_chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness])

#Streamlit implementation
sl.header("Lixibox CS bot")
query=sl.text_input('Enter some text')


if(query):
        with tru_recorder as recording:
                response=rag_chain.invoke(query)
        sl.write(response)
        sl.write(retriever.invoke(query))
                
tru.run_dashboard(port = 8502)