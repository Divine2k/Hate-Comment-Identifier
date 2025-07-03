import os
import re
import pickle
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from getpass import getpass
from pypdf import PdfReader
from langchain_community.document_loaders import GithubFileLoader
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent, initialize_agent, AgentType
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser



#STATIC VARIABLES
modelName = "llama-3.1-8b-instant"

language_detect_prompt = """
Detect the language of the following statements and only give the name of the lauguage as output seperated by comas: 
{previous}
{initial}
"""

prompt_template_policy = PromptTemplate(
    input_variables=['policies', 'initial_comment', 'previous_comment'],
    template= """ 
    Analyze this comments specially initial_comment:
    
    previous_comment: {previous_comment}
    initial_comment: {initial_comment} 

    Check if initial_comment strictly follows the policy below (take previous_comment as being referred to by initial_comment).

    policies: {policies}

    IMPORTANT NOTE:
    If 'initial_comment' violates 'policies' then your output should be 'Policy Violated, <State your reason>' or else
    'Policy Not Violated, <State your reason>'.
    Also don't forget to quote the sentence from the 'policies' which is being violated. Keep the reasoning short
    Dont write anything else.
    """
)

prompt_template_semantic = PromptTemplate(
    input_variables=['initial_comment', 'previous_comment'],
    template= """ 
    Analyze this comments specially initial_comment:
    
    previous_comment: {previous_comment}
    initial_comment: {initial_comment} 

    Provide a neutral, concise summary (4-5 lines) of what the initial_comment is expressing in relation to the previous_comment,
    avoiding any reproduction of potentially offensive or aggressive language.
    """
)

model_kwargs = {'device': 'cpu'}

prev_comment = input('Enter the comment which has been addressed: ')
init_comment = input('Enter the comment which needes to be checked: ')

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def text_pre_process(comment) -> str:
    cleanComment = re.sub(r"<.*?>", "", comment)
    cleanComment = cleanComment.strip()
    return cleanComment

def construct_message_dict(prev, initial) -> dict:
    return {
        'previous_comment': prev,
        'previous_comment_lang': '',
        'initial_comment': initial,
        'initial_comment_lang': '',
        'initial_comment_analysis_summary': '',
        'overall_summary': ''
    }
    
def initialize_model():
    llm = ChatGroq(
        model=modelName,
        temperature=0.34,
        max_tokens=500,
        timeout=None,
        max_retries=2,
        api_key='YOUR API KEY'
        )
    return llm

def parse_pdf(file, number_of_pages):
    joined_text = []
    for page_number in range(number_of_pages):
        page = file.pages[page_number]
        text = page.extract_text()
        joined_text.append(text)
    return joined_text

def preprocess_doc(raw) -> str:
    clean_communityG = re.sub(r"|\n|\n1|\n2|\uffff", "", raw)
    return clean_communityG

def init_text_splitter():
    t = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=5,
        length_function=len,
        is_separator_regex=False,
    )
    return t
    
def init_embedding_model():
    embedding_model = HuggingFaceEmbeddings(    
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs = model_kwargs
        )
    return embedding_model

def init_vectorDB(embedding_model):
    vector_store = Chroma(
        collection_name="code_collection",
        embedding_function=embedding_model,
        persist_directory="vectorDB_Code", 
        )
    return vector_store

clean_prev_comment = text_pre_process(prev_comment)
clean_init_comment = text_pre_process(init_comment)

comment_constructor = construct_message_dict(clean_prev_comment, clean_init_comment)

prompt_LD = PromptTemplate(
    input_variables=["previous", 'initial'], template = language_detect_prompt
)

llm = initialize_model()

lang_detect_chain = prompt_LD | llm | StrOutputParser()
lang_detect_result = lang_detect_chain.invoke({'previous': comment_constructor['previous_comment'],
                                               'initial': comment_constructor['initial_comment']}
                                  )

comment_constructor['previous_comment_lang'] = lang_detect_result.split(',')[0]
comment_constructor['initial_comment_lang'] = lang_detect_result.split(',')[1]

#PDF Parsing
reader = PdfReader("community guidelines.pdf")
number_of_pages = len(reader.pages)

joined_text = parse_pdf(reader, number_of_pages)
community_policies_raw = ' '.join(joined_text)
community_policies_clean = preprocess_doc(community_policies_raw)

#Setting up VectorDB
policies = [Document(page_content=community_policies_clean)]
text_splitter = init_text_splitter()

split_communityG = text_splitter.split_documents(policies)

embedding_model = init_embedding_model()
vector_store = init_vectorDB(embedding_model)

# uuids = [str(uuid4()) for _ in range(len(split_communityG))]
# vector_store.add_documents(documents=split_communityG, ids=uuids)
retriver = vector_store.as_retriever()
db_result = vector_store.similarity_search(comment_constructor['initial_comment'], k=3)

chain_policy = prompt_template_policy | llm | StrOutputParser()
chain_policy_output = chain_policy.invoke({
                'policies': db_result[0].page_content,
                'initial_comment': comment_constructor['initial_comment'],
                'previous_comment': comment_constructor['previous_comment'],
})

chain_semantic = prompt_template_semantic | llm | StrOutputParser()
chain_semantic_output = chain_semantic.invoke({
                'initial_comment': comment_constructor['initial_comment'],
                'previous_comment': comment_constructor['previous_comment'],
})

comment_constructor['initial_comment_analysis_summary'] = chain_policy_output
comment_constructor['overall_summary'] = chain_semantic_output

print(comment_constructor)










