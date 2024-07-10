from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
llm = ChatOpenAI(temperature=0, max_tokens=500)

template_S = "You are a helpful assistant that help user find visited sites from browser history."
system_message_prompt = SystemMessagePromptTemplate.from_template(template_S)

template = """Generate short summary from provided context with given:\n{date}.\nand provide {url} in reply

{context}
If the context is empty reply "kindly adjust your search for better result"
"""
prompt = ChatPromptTemplate.from_messages([system_message_prompt, template])

embedding_function = HuggingFaceBgeEmbeddings(
            model_name = 'all-MiniLM-L6-v2',
            model_kwargs = {'device':'cpu'},
            encode_kwargs = {'normalize_embeddings':True}
        )

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content={self.page_content}, metadata={self.metadata})"


class Core:
    def makeDocs(self,docs):
        print('recursive splitter')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 400, chunk_overlap = 100)
        docs = text_splitter.split_documents(docs)
        print('db')
        db = FAISS.from_documents(docs, embedding_function)
        print('parent child splitter')
        parent_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=100)
        child_splitter = CharacterTextSplitter(separator="\n", chunk_size=400, chunk_overlap=50)

        store = InMemoryStore() 
        print('parent doc retriever')
        par_doc_retriever = ParentDocumentRetriever(vectorstore=db, docstore=store, child_splitter=child_splitter, parent_splitter=parent_splitter)
        par_doc_retriever.add_documents(docs)

        return par_doc_retriever
    
    def LLMResponse(self):
        chain = (
            {"context": RunnablePassthrough(), "url":RunnablePassthrough(), "date":RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain
