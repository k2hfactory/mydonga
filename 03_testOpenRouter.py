import os
import sys

# Windows 터미널 출력 인코딩 설정 (Emoji 및 UTF-8 지원)
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# .env파일 로드
load_dotenv()

# OpenRouter API 모델 설정
api_key = os.getenv('OPENROUTER_API_KEY')
model = ChatOpenAI(
    model          = "openai/gpt-4o-mini",        
    openai_api_key = api_key,
    base_url       = "https://openrouter.ai/api/v1" 
)




#2 프롬프트 템플릿
prompt = ChatPromptTemplate([
   ("system", "당신은 친절하고 전문적인 인공지능 AI선생님이야. 사용자의 질문에 한국어로 친절히 답해주세요."),
   ("user", "{ask}에 대해서 설명해줘")
  ])

#3 LCEL언어지원  | 연결
print('OpenRouter사용  test')
chain = prompt | model | StrOutputParser()
result = chain.invoke({'ask':'제주도'}) 
print(result)



