import os
import sys
import json
import math
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
try:
    from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
except ImportError:
    from langchain.agents import AgentExecutor, create_openai_tools_agent

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

# Windows 터미널 출력 인코딩 설정 (UTF-8 지원)
sys.stdout.reconfigure(encoding='utf-8')

# .env 환경 변수 로드
load_dotenv()


# =========================================================
# 0. 처리 결과 JSON 저장 함수 (data2/jejumath.json)
# =========================================================
def save_to_jejumath_json(question: str, response: str):
    """처리 결과 내용을 data2 폴더의 jejumath.json 파일에 기록 및 저장하는 함수"""
    dir_path = "data2"
    file_path = os.path.join(dir_path, "jejumath.json")
    
    os.makedirs(dir_path, exist_ok=True)

    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "response": response
    }

    records = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    records.append(new_entry)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# =========================================================
# 1. Pydantic BaseModel 상속 입력 스키마 정의
# =========================================================
class MathQuery(BaseModel):
    """수학 연산 및 파이썬 내장/math 함수 정보를 정의하는 입력 스키마"""
    operation: str = Field(
        ..., 
        description="수행할 연산 종류 ('add', 'subtract', 'multiply', 'divide', 'abs', 'round', 'sqrt', 'pow')"
    )
    num1: float = Field(..., description="첫 번째 숫자 (또는 대상 숫자)")
    num2: float = Field(default=0.0, description="두 번째 숫자 (지수, 소수점 자릿수, 뺄셈/나눗셈 인자 등)")

    def calculate(self) -> float:
        """파이썬 내장 함수 및 math 모듈 함수를 참조하여 수학 연산을 수행하는 메서드"""
        op = self.operation.lower()

        # 파이썬 내장 함수 및 math 모듈 참조 딕셔너리
        math_funcs = {
            'abs': abs,
            'round': round,
            'sqrt': math.sqrt,
            'pow': math.pow
        }

        if op in math_funcs:
            func = math_funcs[op]
            if op == 'sqrt':
                return func(self.num1)
            elif op == 'abs':
                return func(self.num1 - self.num2) if self.num2 != 0.0 else func(self.num1)
            elif op == 'round':
                return func(self.num1, int(self.num2))
            elif op == 'pow':
                return func(self.num1, self.num2)
        elif op == "add":
            return self.num1 + self.num2
        elif op == "subtract":
            return self.num1 - self.num2
        elif op == "multiply":
            return self.num1 * self.num2
        elif op == "divide":
            if self.num2 == 0:
                raise ValueError("0으로 나눌 수 없습니다.")
            return self.num1 / self.num2
        else:
            raise ValueError(f"지원하지 않는 연산 타입입니다: {self.operation}")


class JejuQuery(BaseModel):
    """제주도 정보(날씨, 관광지, 특산물/맛집, 여행팁) 조회를 위한 입력 스키마"""
    category: str = Field(
        ..., 
        description="조회할 제주 정보 카테고리 ('weather', 'tourist_spot', 'food', 'tip')"
    )
    location: str = Field(default="제주도 전체", description="조회할 제주 세부 지역 (예: 서귀포, 애월, 성산, 제주시)")
    date: Optional[str] = Field(default="today", description="조회할 날짜 (예: 오늘, 내일, YYYY-MM-DD)")

    def get_jeju_info(self) -> str:
        """제주 요청 카테고리에 맞는 요약 정보를 반환하는 메서드"""
        if self.category == "weather":
            return f"🌤️ [{self.location}] {self.date} 날씨: 맑음, 기온: 22°C (여행하기 좋은 날씨입니다)"
        elif self.category == "tourist_spot":
            return f"🌋 [{self.location}] 대표 추천 관광지: 성산일출봉, 섭지코지, 한라산 국립공원, 곽지해수욕장"
        elif self.category == "food":
            return f"🍊 [{self.location}] 추천 특산물 및 맛집: 흑돼지 구이, 제주 감귤/한라봉, 고기국수, 갈치조림"
        elif self.category == "tip":
            return f"💡 [{self.location}] 제주 여행 팁: 렌터카 사전 예약 필수, 해안도로 드라이브 추천, 일몰 시간 확인"
        else:
            return f"🏝️ [{self.location}] 제주도 가이드 정보 제공 완료"


# =========================================================
# 2. @tool 데코레이터 적용 (args_schema 인자 활용)
# =========================================================
@tool(args_schema=MathQuery)
def math_tool(operation: str, num1: float, num2: float = 0.0) -> str:
    """수학 계산 및 내장 함수(abs, round, sqrt, pow 및 사칙연산 add, subtract, multiply, divide)를 수행하는 툴입니다."""
    query = MathQuery(operation=operation, num1=num1, num2=num2)
    result = query.calculate()
    return f"계산 결과 ({operation}): {result}"


@tool(args_schema=JejuQuery)
def jeju_tool(category: str, location: str = "제주도 전체", date: str = "today") -> str:
    """제주도의 날씨, 관광지, 특산물/맛집, 여행 팁 정보를 제공하는 전용 툴입니다."""
    query = JejuQuery(category=category, location=location, date=date)
    return query.get_jeju_info()


# =========================================================
# 3. AgentExecutor 파이프라인 생성 함수
# =========================================================
def create_agent_pipeline():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY가 .env 파일에 설정되지 않았습니다.")

    tools = [math_tool, jeju_tool]

    model = ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 제주도 여행 정보 및 수학 연산 도구를 자율적으로 선택하여 사용자의 질문에 정확하게 답변하는 친절한 AI 가이드입니다."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(model, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
    )
    return executor


# =========================================================
# 4. 메인 실행 함수 (CLI 터미널 실행)
# =========================================================
def process_query(agent_executor, user_input: str, chat_history: list):
    """사용자 질문을 처리하고 결과를 출력 및 JSON 저장"""
    print(f"\n==================================================")
    print(f"👤 사용자 질문: {user_input}")
    print(f"==================================================")

    res = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    final_output = res.get("output", "")
    intermediate_steps = res.get("intermediate_steps", [])

    step_logs = []
    if intermediate_steps:
        print("\n⚙️ [도구 실행 과정 (Intermediate Steps)]")
        for action, tool_output in intermediate_steps:
            log_str = f"✅ [{action.tool} 호출] 인자: {action.tool_input} ➔ 실행 결과: {tool_output}"
            print(log_str)
            step_logs.append(log_str)

    if step_logs:
        formatted_response = "\n".join(step_logs) + "\n\n" + final_output
    else:
        formatted_response = final_output

    print(f"\n🤖 AI 최종 답변:\n{final_output}")

    # data2/jejumath.json 저장
    save_to_jejumath_json(user_input, formatted_response)
    print(f"\n💾 처리 결과가 'data2/jejumath.json'에 저장되었습니다.")

    # 세션 기록에 추가
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=final_output))


if __name__ == "__main__":
    print("🚀 mymathjeju.py AgentExecutor CLI 애플리케이션을 시작합니다.")
    
    try:
        executor = create_agent_pipeline()
        chat_history = []

        # 테스트 질문 목록 (수학 내장 함수 abs, round, sqrt, pow 및 제주도 가이드)
        test_queries = [
            "제주도 서귀포 특산물 및 맛집 알려줘",
            "abs(2 - 17) 계산해줘",
            "sqrt(16) 연산해줘",
            "pow(2, 5) 계산해줘",
            "round(3.14159, 2) 계산해줘"
        ]

        for q in test_queries:
            process_query(executor, q, chat_history)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
