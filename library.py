from langchain_openai import ChatOpenAI
from langchain_classic.prompts import PromptTemplate
from langchain_classic.embeddings import OpenAIEmbeddings
from langchain_classic.chains import RetrievalQA, LLMChain, SequentialChain
from langchain_classic.document_loaders import PyPDFLoader
from langchain_community.document_loaders import GitLoader
from langchain_classic.vectorstores import FAISS

import os
from dotenv import load_dotenv

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

##################################################################################################################################

class InsightNavigator:
    """인사이트 추출 및 분석을 위한 기본 클래스"""

    def __init__(self, documents, llm_model="gpt-4o-mini"):
        # 문서 리스트 저장
        self.documents = documents
        # 임베딩 설정
        self.embeddings = OpenAIEmbeddings()
        # 벡터 저장소 생성
        self.db = FAISS.from_documents(documents, self.embeddings)
        # LLM 모델 설정
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0)
        # 질의응답을 위한 QA 체인 설정
        self.qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.db.as_retriever())
        
        # 요약을 위한 프롬프트 및 체인 설정
        summary_prompt = PromptTemplate(input_variables=['sentence'], 
            template="""다음 문장의 핵심 내용을 요약해주세요.
            {sentence}
            """
        )
        summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt, output_key="summary")
        
        # 번역을 위한 프롬프트 및 체인 설정
        translate_prompt = PromptTemplate.from_template("다음 문장을 한글로 번역하세요.\n{summary}")
        translate_chain = LLMChain(llm=self.llm, prompt=translate_prompt, output_key="translation")
        
        # 번역 후 요약하는 순차적 체인 구성
        self.chains = SequentialChain(
            chains=[summary_chain, translate_chain], 
            input_variables=["sentence"], 
            output_variables=["summary", "translation"]
        )

    def run(self, text=None):
        """텍스트를 번역하고 요약합니다."""
        if text is None:
            # 전달된 텍스트가 없으면 모든 문서 내용을 합침
            text = " ".join([doc.page_content for doc in self.documents])
        
        # 텍스트가 너무 길면 토큰 제한을 고려하여 앞부분 일부만 사용
        # text = text[:4000]
        
        # 체인 실행
        result = self.chains({"sentence": text})
        return result

    def query(self, text):
        result = self.qa({"query": text})
        return result["result"]

##################################################################################################################################

class PdfNavigator(InsightNavigator):
    """PDF 파일 분석을 위한 클래스"""

    def __init__(self, pdf_file, llm_model="gpt-4o-mini"):
        # PDF 문서를 로드한 후 부모 클래스 초기화
        super().__init__(self._load_(pdf_file), llm_model)

    def _load_(self, file):
        # PDF 파일 로딩
        loader = PyPDFLoader(file)
        docs = loader.load()
        return docs

##################################################################################################################################

class GithubNavigator(InsightNavigator):
    """Github 레포지토리 분석을 위한 클래스"""

    def __init__(self, url, path, branch="main", llm_model="gpt-4o-mini"):
        # Github 문서를 로드한 후 부모 클래스 초기화
        super().__init__(self._load_(url, path, branch), llm_model)

    def _load_(self, url, path, branch="main"):
        # Git 레포지토리 로딩 (마크다운 파일만 필터링)
        loader = GitLoader(clone_url=url, repo_path=path, branch=branch, file_filter=GithubNavigator.file_filter)
        raw_docs = loader.load()
        return raw_docs

    @staticmethod
    def file_filter(file_path):
        # .md 확장자 파일만 대상
        return file_path.endswith(".md")
