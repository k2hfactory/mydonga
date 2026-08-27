from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini",  temperature=0)
# model = ChatOpenAI(model="gpt-4o")

@tool
def get_weather(city:str) ->str:
  """ 제주도 날씨 조회 """
  return f"{city} 지역 날씨"  


tools = [get_weather]
tool_dict ={'get_weather' : get_weather}
llm_with_tools = model.bind_tools(tools)

messages = [
    SystemMessage("당신은 사용자의 질문에 답변을 하기 위해 tools를 사용할 수 있다."),
    HumanMessage("제주도 날씨 상황 알려줘")
]                        

response = llm_with_tools.invoke(messages)
messages.append(response)

print(response.tool_calls) #처음실행해서 출력정보 좋아요
print()

for tool_call in response.tool_calls:
  selected_tool = tool_dict.get(tool_call['name'])
  tool_msg = selected_tool.invoke(tool_call)
  messages.append(tool_msg)

final_reponse = llm_with_tools.invoke(messages)
print(final_reponse.content)

print()
print("2번째수식처리 @tool적용")
@tool
def calculator(expression: str) ->str:
  """
  수식을 계산합니다
  예: 10+20, 30*5
  """
  return str(eval(expression))

print('연산결과 ', calculator.invoke({"expression": "1200+79"}))
print()


print("3번째수식처리 @tool적용")
@tool
def add(a:int , b:int) ->int:
  """두 숫자를 더함"""
  return a+b

@tool
def multiply(a: int, b: int) -> int:
    """두 숫자를 곱함"""
    return a * b

print('더하기결과 =', add.invoke({"a": 5, "b": 3}))
print('곱하기결과 =', multiply.invoke({"a": 5, "b": 3}))
