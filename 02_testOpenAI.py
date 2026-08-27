import sys

# Windows 터미널 출력 인코딩 설정 (Emoji 및 UTF-8 지원)
sys.stdout.reconfigure(encoding='utf-8')

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# .env파일 로드
load_dotenv()

#1 모델기술
model = ChatOpenAI(model='gpt-4o-mini')

# 더권장
# api_key = os.getenv('OPENROUTER_API_KEY')
# model = ChatOpenAI(
#     model          = "openai/gpt-4o-mini",        
#     openai_api_key = api_key,
#     base_url       = "https://openrouter.ai/api/v1" 
# )


#2 프롬프트 템플릿
prompt = ChatPromptTemplate([
   ("system", "당신은 친절하고 전문적인 인공지능 AI선생님이야. 사용자의 질문에 한국어로 친절히 답해주세요."),
   ("user", "{ask}에 대해서 설명해줘")
  ])

#3 LCEL언어지원  | 연결
chain = prompt | model | StrOutputParser()
result = chain.invoke({'ask':'케데헌 스토리 상세하게 설명 주인공및 장소 설명'}) 
print(result)


