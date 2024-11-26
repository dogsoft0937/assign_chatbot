from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from langchain_core.messages import ChatMessage

class ChatLLM:
    def __init__(self):
        self.model = ChatOllama(model = 'gemma2:2b',temperature=3)

        self._template = """주어진 질문에 짧고 간결하게 한글로 답변을 제공해주세요
        
         Question: {question}
        """

        self._prompt = ChatPromptTemplate.from_template(template=self._template)

        self._chain = (
            {'question' : RunnablePassthrough()} | self._prompt | self.model | StrOutputParser()
        )

    def invoke(self,user_input):
        response = self._chain.invoke(user_input)
        return response
    
    def format_docs(self,docs):
        return '\n\n'.join([d.page_content for d in docs])
class ChatWeb:
    def __init__(self,llm,page_title="Gazzi Chatbot",page_icon=':books:'):
        self._page_title=page_title
        self._page_icon=page_icon
        self._llm=llm
    def print_messages(self):
        if "messages" in st.session_state and len(st.session_state['messages'])>0:
            for chat_message in st.session_state['messages']:
                st.chat_message(chat_message.role).write(chat_message.content)
    def run(self):
        st.set_page_config(page_title=self._page_title,page_icon=self._page_icon)
        st.title(self._page_title)

        if "messages" not in st.session_state:
            st.session_state['messages']=[]
        self.print_messages()

        if user_input:=st.chat_input("질문을 입력해 주세요."):
            st.chat_message("user").write(f"{user_input}")
            st.session_state['messages'].append(ChatMessage(role='user',content=user_input))
            response = self._llm.invoke(user_input)
            with st.chat_message("assistant"):
                msg_assistant = response
                # msg_assistant= f"당신이 입력한 내용:{user_input}"
                st.write(msg_assistant)
                st.session_state['messages'].append(ChatMessage(role='assistant',content=msg_assistant))


if __name__ =='__main__':
    llm=ChatLLM()
    web=ChatWeb(llm=llm)
    web.run()