import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings


class PDFKnowledgeBaseQA:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

        os.environ["DASHSCOPE_API_KEY"] = "sk-2583d9d000d642e98254164d7aeb532d"

        self.embeddings = DashScopeEmbeddings(model="text-embedding-v1")

        self.llm = Tongyi(model_name="qwen-turbo", temperature=0)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        self.vectorstore = None
        self.qa_chain = None

    def build_vectorstore(self):
        """构建向量数据库"""
        print("正在加载PDF...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        print(f"正在分割文档...")
        split_docs = self.text_splitter.split_documents(documents)

        print(f"正在创建向量数据库...")
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        print("向量数据库创建完成！")

    def create_qa_chain(self):
        """创建问答链"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        prompt_template = """基于以下已知信息回答问题。如果无法找到答案，请说"根据PDF文档无法回答该问题"。

已知信息：{context}
问题：{question}
答案："""

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )
        print("问答链创建完成！")

    def ask(self, question: str):
        """提问"""
        result = self.qa_chain({"query": question})
        print(f"\n问题：{question}")
        print(f"答案：{result['result']}")


# 使用
if __name__ == "__main__":
    qa = PDFKnowledgeBaseQA(pdf_path=r"D:\Users\Esther\Desktop\语言模型基础.pdf")
    qa.build_vectorstore()
    qa.create_qa_chain()
    qa.ask("什么是大语言模型？")
