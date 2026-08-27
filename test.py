import os
import sys
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

# Windows 터미널 출력 인코딩 설정 (UTF-8 지원)
sys.stdout.reconfigure(encoding='utf-8')

# .env 환경 변수 로드
load_dotenv()


# 1. Pydantic BaseModel 상속 입력 스키마 정의
class MathQuery(BaseModel):
    """수학 연산 정보를 정의하는 입력 스키마"""
    operation: str = Field(
        ..., 
        description="수행할 연산 종류 ('add', 'subtract', 'multiply', 'divide', 'abs')"
    )
    num1: float = Field(..., description="첫 번째 숫자")
    num2: float = Field(default=0.0, description="두 번째 숫자")

    def calculate(self) -> float:
        """연산을 수행하고 결과를 반환하는 메서드"""
        if self.operation == "add":
            return self.num1 + self.num2
        elif self.operation == "subtract":
            return self.num1 - self.num2
        elif self.operation == "multiply":
            return self.num1 * self.num2
        elif self.operation == "divide":
            if self.num2 == 0:
                raise ValueError("0으로 나눌 수 없습니다.")
            return self.num1 / self.num2
        elif self.operation == "abs":
            return abs(self.num1 - self.num2)
        else:
            raise ValueError(f"지원하지 않는 연산 타입입니다: {self.operation}")


class WeatherQuery(BaseModel):
    """날씨 조회 정보를 정의하는 입력 스키마"""
    location: str = Field(..., description="조회할 도시 또는 지역 이름 (예: 서울, 제주도)")
    date: Optional[str] = Field(default="today", description="조회할 날짜 (예: 오늘, 내일, YYYY-MM-DD)")
    unit: str = Field(default="celsius", description="온도 단위 ('celsius' 또는 'fahrenheit')")

    def get_info(self) -> str:
        """날씨 요청 요약 문자열을 반환하는 메서드"""
        unit_str = "섭씨(°C)" if self.unit == "celsius" else "화씨(°F)"
        return f"[{self.location}] {self.date} 날씨 정보 (단위: {unit_str}, 상태: 맑음, 기온: 24{unit_str})"


# 2. @tool 데코레이터 적용 (args_schema 인자 활용)
@tool(args_schema=MathQuery)
def math_tool(operation: str, num1: float, num2: float = 0.0) -> str:
    """수학 계산을 수행하는 툴입니다. ('add', 'subtract', 'multiply', 'divide', 'abs' 연산 가능)"""
    query = MathQuery(operation=operation, num1=num1, num2=num2)
    result = query.calculate()
    return f"계산 결과 ({operation}): {result}"


@tool(args_schema=WeatherQuery)
def weather_tool(location: str, date: str = "today", unit: str = "celsius") -> str:
    """지역 및 날짜별 날씨 정보를 조회하는 툴입니다."""
    query = WeatherQuery(location=location, date=date, unit=unit)
    return query.get_info()


# 3. 툴 리스트 및 딕셔너리 구성
tools = [math_tool, weather_tool]
tools_dict = {t.name: t for t in tools}


# ---------------------------------------------------------
# Streamlit UI 및 세션(Session State) 관리
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI 스마트 어시스턴트 (Session 세션 관리)",
    page_icon="🤖",
    layout="wide"
)

# 세션 상태 초기화 (st.session_state)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 사이드바 (Sidebar) 구성
st.sidebar.title("⚙️ 세션 & 모델 설정")
st.sidebar.markdown("---")

# OpenRouter API 키 상태 확인
api_key = os.getenv("OPENROUTER_API_KEY")
if api_key:
    st.sidebar.success("🔑 OpenRouter API 연결됨")
else:
    st.sidebar.error("⚠️ OPENROUTER_API_KEY를 .env에서 찾을 수 없습니다.")

# 대화 세션 초기화 버튼
if st.sidebar.button("🗑️ 대화 기록 초기화 (Clear Session)", use_container_width=True):
    st.session_state["messages"] = []
    st.rerun()

st.sidebar.markdown("---")
# 현재 세션 메시지 수 표시
st.sidebar.metric("💬 현재 세션 대화 수", f"{len(st.session_state['messages'])}개")

# 모델 파라미터 설정
temperature = st.sidebar.slider("Temperature (창의성)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("💡 빠른 질문 예시")
preset_query = ""
if st.sidebar.button("🌤️ 서울 오늘 날씨 알려줘", use_container_width=True):
    preset_query = "서울 오늘 날씨 알려줘"
if st.sidebar.button("🔢 abs(2 - 17) 계산해줘", use_container_width=True):
    preset_query = "abs(2 - 17) 계산해줘"
if st.sidebar.button("🏝️ 제주도 내일 날씨 알려줘", use_container_width=True):
    preset_query = "제주도 내일 날씨 알려줘"


# LCEL 모델 및 세션 메모리 지원 파이프라인 초기화
@st.cache_resource
def get_lcel_chain(temp: float):
    model = ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=temp
    )
    model_with_tools = model.bind_tools(tools)
    
    # MessagesPlaceholder를 포함하여 대화 히스토리 context 전달
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 도구(Tool)를 사용하여 사용자의 질문에 정확하게 답하는 AI 어시스턴트입니다."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}")
    ])
    
    def execute_tool_calls(ai_message) -> str:
        if not ai_message.tool_calls:
            return ai_message.content

        results = []
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            target_tool = tools_dict.get(tool_name)
            if target_tool:
                output = target_tool.invoke(tool_args)
                results.append(f"✅ [{tool_name} 호출 성공]\n- 인자: `{tool_args}`\n- 실행 결과: **{output}**")
            else:
                results.append(f"❌ [{tool_name}] 존재하지 않는 툴입니다.")
        return "\n\n".join(results)

    return prompt | model_with_tools | execute_tool_calls


lcel_chain = get_lcel_chain(temperature)

# 메인 화면 구성
st.title("🤖 AI 스마트 어시스턴트 (Session Memory)")
st.caption("Streamlit `st.session_state` 세션 관리 및 LangChain LCEL 툴 호출 파이프라인")

# 저장된 세션 대화 기록 메인 화면에 출력
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 받기
user_input = st.chat_input("질문을 입력하세요...")
if preset_query:
    user_input = preset_query

if user_input:
    # 1. 사용자 질문을 세션 상태에 추가 및 화면 출력
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. 세션 대화 기록을 LangChain 메시지 객체로 변환 (Multi-turn Context)
    chat_history = []
    for m in st.session_state["messages"][:-1]:  # 현재 질문 이전 메시지들
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            chat_history.append(AIMessage(content=m["content"]))

    # 3. AI 답변 생성 및 세션 업데이트
    with st.chat_message("assistant"):
        with st.spinner("세션 대화 히스토리와 툴을 적용하여 답변을 생성 중입니다..."):
            try:
                response = lcel_chain.invoke({
                    "question": user_input,
                    "chat_history": chat_history
                })
                st.markdown(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"❌ 오류가 발생했습니다: {str(e)}"
                st.error(error_msg)
                st.session_state["messages"].append({"role": "assistant", "content": error_msg})
