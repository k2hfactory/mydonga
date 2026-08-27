import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import pytz
import yfinance as yf
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Windows 콘솔 한글 인코딩 설정
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 환경변수 로드 (.env의 OPENAI_API_KEY)
load_dotenv()

# 모델 설정
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# model = ChatOpenAI(model="gpt-4o", temperature=0)


# ==========================================
# 1. Pydantic을 이용한 Tool 인자 스키마 정의
# ==========================================

class StockHistoryInput(BaseModel):
    """주식 종목 가격 데이터 조회를 위한 입력 스키마"""
    ticker: str = Field(..., title="주식코드", description="조회할 주식의 티커 심볼 (예: AAPL, TSLA, 005930.KS)")
    period: str = Field(..., title="조회기간", description="데이터 조회 기간 (예: 1d, 5d, 1mo, 1y, 5y, max)")


class CurrentTimeInput(BaseModel):
    """현재 시간 조회를 위한 입력 스키마"""
    timezone: str = Field(..., title="타임존", description="IANA 타임존 문자열 (예: Asia/Seoul, America/New_York, Europe/London)")
    location: str = Field(..., title="지역명", description="해당 타임존의 지역/도시 이름 (예: 서울, 뉴욕, 런던)")


# ==========================================
# 2. @tool 데코레이터와 args_schema 연결
# ==========================================

@tool(args_schema=StockHistoryInput)
def get_yf_stock_history(ticker: str, period: str) -> str:
    """주식 종목의 과거 가격 및 거래량 데이터를 조회하여 마크다운 표 형식으로 반환하는 함수"""
    try:
        stock = yf.Ticker(ticker=ticker)
        history = stock.history(period=period)
        
        if history.empty:
            return f"티커 '{ticker}'에 대한 기간 '{period}'의 주가 데이터를 찾을 수 없습니다."
        
        # pandas의 to_markdown()은 'tabulate' 라이브러리가 설치되어 있어야 동작합니다.
        history_md = history.to_markdown()
        return history_md
    except Exception as e:
        return f"주식 데이터 조회 중 오류 발생: {str(e)}"


@tool(args_schema=CurrentTimeInput)
def get_current_time(timezone: str, location: str) -> str:
    """지정된 타임존과 지역의 현재 날짜 및 시간을 YYYY-MM-DD HH:MM:SS 형식으로 반환하는 함수"""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"{timezone} ({location}) 현재 시간: {now}"
    except Exception as e:
        return f"시간 조회 중 오류 발생: {str(e)}"


# ==========================================
# 3. 도구(Tools) 목록 구성 및 모델 바인딩
# ==========================================

tools = [get_yf_stock_history, get_current_time]
tool_dict = {tool.name: tool for tool in tools}

# LLM에 도구 바인딩
llm_with_tools = model.bind_tools(tools)


# ==========================================
# 4. 사용자 질의 실행 및 도구 호출 처리
# ==========================================

messages = [
    SystemMessage("당신은 사용자의 질문에 정확하게 답변하기 위해 제공된 도구(tools)를 적극적으로 활용하는 AI 어시스턴트입니다."),
    HumanMessage("테슬라(TSLA)의 최근 3일간 주가 정보를 조회해서 요약해줘.")
]

print(">>> [1단계] 사용자 질문 전송 및 LLM 1차 응답(Tool Call 요청)...")
response = llm_with_tools.invoke(messages)
messages.append(response)

# LLM이 도구 사용을 요청(tool_calls)했는지 확인
if response.tool_calls:
    print(f"-> 호출된 도구 개수: {len(response.tool_calls)}")
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"-> 실행할 도구: {tool_name}, 전달 인자: {tool_args}")
        
        selected_tool = tool_dict.get(tool_name)
        if selected_tool:
            tool_msg = selected_tool.invoke(tool_call)
            messages.append(tool_msg)
        else:
            print(f"[경고] 등록되지 않은 도구 호출: {tool_name}")

    print("\n>>> [2단계] 도구 실행 결과를 바탕으로 LLM 최종 답변 생성...")
    final_response = llm_with_tools.invoke(messages)
    print("\n" + "=" * 50)
    print("[최종 답변]")
    print("=" * 50)
    print(final_response.content)
else:
    print("\n" + "=" * 50)
    print("[도구 호출 없이 생성된 답변]")
    print("=" * 50)
    print(response.content)


# ==========================================
# 필요한 라이브러리 설치 안내
# ==========================================
# pip install yfinance pytz tabulate pydantic langchain-core langchain-openai python-dotenv
