# ── 패키지 설치 안내 ──────────────────────────────────────────
# pip uninstall whisper -y
# pip install openai-whisper torch

import sys
import os
import time
import warnings

# ── Windows 콘솔 UTF-8 인코딩 설정 (Emoji 출력 방지) ─────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ── PyTorch & Whisper CPU FP16 경고 전역 억제 ──────────────────
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*FP16 is not supported on CPU.*"
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*You are using `torch.load` with `weights_only=False`.*"
)


#  1단계 : whisper 패키지 유효성 및 모듈 검사
def validate_whisper():
    """
    openai-whisper 패키지 유효성 검사
    (잘못된 'whisper' 패키지 또는 미설치 상태 구분 감지)
    """
    try:
        import whisper

        # ── 검사 1: load_model 함수 존재 여부 확인 ────────
        if not hasattr(whisper, "load_model"):
            raise ImportError(
                "잘못된 whisper 패키지가 설치되어 있습니다.\n"
                "다음 명령어로 올바른 openai-whisper 패키지를 설치하세요:\n"
                "  pip uninstall whisper -y\n"
                "  pip install openai-whisper"
            )

        # ── 검사 2: 단일 .py 파일 섀도잉 여부 확인 ───────
        whisper_file = getattr(whisper, "__file__", None)
        if whisper_file is not None and str(whisper_file).endswith("whisper.py"):
            raise ImportError(
                "현재 디렉토리에 단일 'whisper.py' 파일이 존재하여 모듈 출처가 충돌합니다.\n"
                "해당 파일명을 다른 이름으로 변경해주세요."
            )

        print("✅ whisper 패키지 정상 확인 완료")
        return whisper

    except ModuleNotFoundError:
        print("❌ [패키지 미설치 오류]: 'whisper' 모듈을 찾을 수 없습니다.")
        print("💡 터미널에서 다음 명령어를 실행하여 라이브러리를 설치하세요:")
        print("   pip install openai-whisper torch")
        sys.exit(1)
    except ImportError as e:
        print(f"❌ [패키지 가져오기 오류]: {e}")
        sys.exit(1)
    except Exception as ex:
        print(f"❌ [예상치 못한 오류]: {ex}")
        sys.exit(1)


#  2단계 : 디바이스 감지 (GPU / CPU 자동 선택)
def detect_device() -> tuple[str, bool]:
    """
    GPU (CUDA) / CPU 자동 감지 및 FP16 연산 설정
    Returns:
        device   : "cuda" 또는 "cpu"
        use_fp16 : GPU=True / CPU=False
    """
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    # CPU 실행 시 FP16 미지원이므로 False 지정 (경고 방지)
    use_fp16 = (device == "cuda")

    print(f"  🖥️  실행 장치 : {device.upper()}")
    print(f"  ⚡ FP16 연산  : {'활성화 (GPU)' if use_fp16 else '비활성화 (CPU → FP32)'}")

    return device, use_fp16


#  3단계 : Whisper 모델 로딩
def load_whisper_model(whisper_module, model_size: str = "base", device: str = "cpu"):
    """Whisper 모델 로딩"""
    print(f"\n🔄 Whisper [{model_size}] 모델 로딩 중... (device={device})")
    start = time.time()

    try:
        model = whisper_module.load_model(model_size, device=device)
        elapsed = time.time() - start
        print(f"✅ 모델 로딩 완료 ({elapsed:.2f}초)")
        return model

    except Exception as e:
        print(f"❌ 모델 로딩 실패: {type(e).__name__}: {e}")
        sys.exit(1)


#  4단계 : 오디오 파일 검증
def validate_audio(audio_path: str) -> str:
    """오디오 파일 존재 여부 및 확장자 검증"""
    supported_exts = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm"}

    # 상대 경로 실행 보장 (현재 디렉토리 및 스크립트 위치 기준 검색)
    target_path = audio_path
    if not os.path.exists(target_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, audio_path)
        if os.path.exists(alt_path):
            target_path = alt_path
        else:
            print(f"❌ 오디오 파일이 존재하지 않습니다: {audio_path}")
            print(f"   현재 작업 디렉토리: {os.getcwd()}")
            print(f"   탐색 시도 경로    : {os.path.abspath(alt_path)}")
            sys.exit(1)

    ext = os.path.splitext(target_path)[1].lower()
    if ext not in supported_exts:
        print(f"❌ 지원하지 않는 오디오 파일 형식입니다: '{ext}'")
        print(f"   지원 형식: {', '.join(sorted(supported_exts))}")
        sys.exit(1)

    file_size_kb = os.path.getsize(target_path) / 1024
    print(f"✅ 오디오 파일 확인: {target_path} ({file_size_kb:.1f} KB)")
    return target_path


#  5단계 : 음성 인식 (전사 Transcription)
def transcribe_audio(
    model,
    use_fp16: bool,
    audio_path: str,
    language: str | None = None
) -> dict:
    """Whisper 음성 → 텍스트 전사"""
    print(f"\n🎙️  음성 인식 시작: {audio_path}")
    print(f"   fp16={use_fp16} | 언어 설정={'자동 감지' if not language else language}")

    start = time.time()

    try:
        options: dict = {"fp16": use_fp16}
        if language:
            options["language"] = language

        result = model.transcribe(audio_path, **options)
        elapsed = time.time() - start
        print(f"✅ 전사 완료 (소요시간: {elapsed:.2f}초)")
        return result
    except FileNotFoundError:
        print("\n❌ [FFmpeg 미설치/경로 오류]: 시스템 PATH에서 'ffmpeg'를 찾을 수 없습니다.")
        print("💡 해결 방법:")
        print("   1) PowerShell (관리자): winget install Gyan.FFmpeg --source winget")
        print("   2) 수동 다운로드: https://www.gyan.dev/ffmpeg/builds/ 후 환경 변수 Path에 bin 디렉토리 추가")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 전사 처리 중 오류 발생: {type(e).__name__}: {e}")
        sys.exit(1)


#  6단계 : 결과 출력
def print_result(result: dict, audio_path: str) -> str:
    """음성 인식 결과 및 세그먼트 타임스탬프 출력"""
    text = result.get("text", "").strip()
    language = result.get("language", "알 수 없음")
    segments = result.get("segments", [])

    print("\n" + "=" * 60)
    print("  📋 Whisper 음성 인식 결과")
    print("=" * 60)
    print(f"  📁 파일 경로  : {audio_path}")
    print(f"  🌐 감지 언어  : {language}")
    print(f"  📝 전사 내용  :\n\n     {text}\n")

    if segments:
        print("  ⏱️  타임스탬프 세그먼트:")
        print("  " + "─" * 56)
        for seg in segments:
            s = seg.get("start", 0)
            e = seg.get("end", 0)
            txt = seg.get("text", "").strip()
            print(f"  [{s:>6.2f}s → {e:>6.2f}s]  {txt}")

    print("=" * 60)
    return text


def main() -> str:
    AUDIO_PATH = "audio/cat.mp3"
    MODEL_SIZE = "base"
    LANGUAGE = "en"  # "en": 영어 / "ko": 한국어 / None: 자동감지

    print("=" * 60)
    print("  🎙️  Whisper 음성 인식 시스템")
    print("=" * 60)

    # 1단계 : 패키지 검증
    whisper_module = validate_whisper()

    # 2단계 : 디바이스 감지 (GPU/CPU)
    device, use_fp16 = detect_device()

    # 3단계 : 모델 로딩
    model = load_whisper_model(whisper_module, MODEL_SIZE, device)

    # 4단계 : 오디오 파일 검증
    validated_path = validate_audio(AUDIO_PATH)

    # 5단계 : 음성 인식
    result = transcribe_audio(
        model=model,
        use_fp16=use_fp16,
        audio_path=validated_path,
        language=LANGUAGE
    )

    # 6단계 : 결과 출력
    return print_result(result, validated_path)


if __name__ == "__main__":
    speech_text = main()
    print(f"\n🔤 최종 추출 텍스트: {speech_text}")

# ── FFmpeg 설치 가이드 ──────────────────────────────────────────
# 첫번째 방법 (수동 다운로드):
# 1. FFmpeg 공식 빌드 사이트 https://www.gyan.dev/ffmpeg/builds/
# 2. ffmpeg-release-essentials.zip 다운로드 후 압축 해제
# 3. 압축 폴더를 C:\ffmpeg-9.0.1-essentials_build\bin 등 원하는 경로에 이동
# 4. 시스템 환경 변수 편집 -> 환경 변수 -> Path -> 새로 만들기: C:\ffmpeg-9.0.1-essentials_build\bin\
# 5. 터미널 확인: ffmpeg -version
#
# 두번째 방법 (winget 이용 자동 설치):
# 1. PowerShell / CMD 관리자 권한으로 실행
# 2. winget install Gyan.FFmpeg --source winget
# 3. 터미널 확인: ffmpeg -version



