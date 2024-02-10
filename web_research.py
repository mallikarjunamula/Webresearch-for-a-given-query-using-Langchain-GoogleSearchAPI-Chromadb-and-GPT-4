import gradio as gr
import langchain
from langchain.retrievers.web_research import WebResearchRetriever
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
import dotenv
dotenv.load_dotenv()
import logging
logging.basicConfig()

def websearch(question):
  # Vectorstore
  vectorstore = Chroma(
      embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai"
  )
  # LLM
  llm = ChatOpenAI(temperature=0)
  # Search
  search = GoogleSearchAPIWrapper()
  # Initialize
  web_research_retriever = WebResearchRetriever.from_llm(
      vectorstore=vectorstore, llm=llm, search=search
  )
  logging
  logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)
  qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
      llm=llm, retriever=web_research_retriever
  )
  langchain.debug = True
  result = qa_chain({"question": question})
  return result

entity = gr.Interface(
    websearch,
    [
      gr.Textbox(label="Query:", value=""),
    ],
    "textbox",
    title="Web Research for a given Query using Langchain and OpenAI's GPT-4",
    theme = "gradio/monochrome"
)
entity.launch()

