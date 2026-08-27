import os
import sys
import ctypes
import warnings

# Windows 콘솔 UTF-8 인코딩 설정
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# PyTorch / Whisper 경고 메시지 억제
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv
import whisper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# .env 파일 로드 (OpenAI / OpenRouter API 키 설정 적용)
load_dotenv()


def play_audio(file_path: str):
    """
    Windows mciSendString API를 활용한 mp3 오디오 재생
    """
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        print(f"❌ 재생할 오디오 파일을 찾을 수 없습니다: {file_path}")
        return

    print(f"🔊 [1단계] 오디오 음성 재생 중: {file_path}")
    try:
        winmm = ctypes.windll.winmm
        # mciSendString 명령으로 MP3 재생 및 재생 완료까지 대기 (wait)
        winmm.mciSendStringW(f'open "{abs_path}" type mpegvideo alias mp3', None, 0, 0)
        winmm.mciSendStringW('play mp3 wait', None, 0, 0)
        winmm.mciSendStringW('close mp3', None, 0, 0)
        print("✅ 오디오 재생 완료\n")
    except Exception as e:
        print(f"⚠️ 오디오 재생 중 오류 발생 (진행 계속): {e}\n")


def transcribe_audio(file_path: str) -> str:
    """
    Whisper 모델을 이용한 음성 -> 텍스트 변환 (STT)
    """
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {abs_path}")

    print(f"🎙️ [2단계] Whisper 모델 기반 음성 텍스트 변환 (STT) 진행 중...")
    model = whisper.load_model("base", device="cpu")
    result = model.transcribe(abs_path, fp16=False)
    text = result.get("text", "").strip()
    print(f"✅ STT 변환 완료! 추출된 텍스트: \"{text}\"\n")
    return text


def main():
    AUDIO_PATH = "audio/cat.mp3"

    # relative path validation
    if not os.path.exists(AUDIO_PATH):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, AUDIO_PATH)
        if os.path.exists(alt_path):
            AUDIO_PATH = alt_path

    print("=" * 60)
    print(" 🐱 LangChain LCEL 기반 오디오 재생 및 음성 텍스트 변환 시스템")
    print("=" * 60 + "\n")

    # 1. 음성 파일 재생 (Play Sound)
    play_audio(AUDIO_PATH)

    # 2. LangChain LCEL 파이프라인 구성
    # STT 함수를 RunnableLambda로 래핑하여 LCEL 체인에 통합
    stt_runnable = RunnableLambda(transcribe_audio)

    # 프롬프트 템플릿 정의
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절하고 전문적인 AI 어시스턴트입니다. 음성에서 인식된 사용자 텍스트(영문 또는 한국어)를 바탕으로 친절하게 한국어로 응답해 주세요."),
        ("user", "다음 음성 인식 텍스트 내용을 바탕으로 요구 사항에 맞추어 정보를 설명해주세요: {audio_text}")
    ])

    # LLM 모델 지정 (ChatOpenAI)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # StrOutputParser 정의
    output_parser = StrOutputParser()

    # LCEL 체인 결합 (| 파이프 연산자 사용)
    # audio_path -> stt_runnable -> prompt -> model -> output_parser
    lcel_chain = (
        {"audio_text": stt_runnable}
        | prompt
        | model
        | output_parser
    )

    # 3. LCEL 파이프라인 실행
    print("⚡ [3단계] LangChain LCEL 파이프라인 실행 (STT -> Prompt -> LLM -> OutputParser)...")
    response = lcel_chain.invoke(AUDIO_PATH)

    print("=" * 60)
    print(" 🤖 LLM 최종 응답 결과")
    print("=" * 60)
    print(response)
    print("=" * 60)


if __name__ == "__main__":
    main()