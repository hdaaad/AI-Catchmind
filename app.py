import io
import math
import random
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from google import genai
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="AI 캐치마인드",
    page_icon="🎨",
    layout="centered",
)

ROUND_SECONDS = 60
QUESTION_COUNT = 5
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 460
MODEL_NAME = "gemini-2.5-flash"
CATEGORIES = ["동물", "과일", "채소", "사물", "교통수단"]

st.markdown(
    """
    <style>
    .block-container {
        max-width: 850px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    div.stButton > button {
        min-height: 3.2rem;
        font-size: 1.05rem;
        font-weight: 700;
        border-radius: 14px;
    }
    .game-card {
        padding: 1rem 1.2rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 18px;
        margin-bottom: 1rem;
    }
    .keyword-box {
        font-size: 1.65rem;
        font-weight: 800;
        text-align: center;
        padding: .8rem;
        border-radius: 16px;
        background: rgba(128,128,128,.10);
        margin-bottom: .7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# 상태 초기화
# -----------------------------
DEFAULT_STATE = {
    "stage": "start",          # start / game / results
    "category": None,
    "questions": [],
    "round_index": 0,
    "results": [],
    "round_started_at": None,
    "snapshot_bytes": None,
    "snapshot_json": None,
    "submitting": False,
    "timeout_handled": False,
    "submit_reason": None,
    "ai_error": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------
# 데이터 / 유틸 함수
# -----------------------------
@st.cache_data
def load_keywords() -> pd.DataFrame:
    csv_path = Path("keyword.csv")
    if not csv_path.exists():
        raise FileNotFoundError("keyword.csv 파일을 GitHub 저장소에 추가해 주세요.")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"카테고리", "키워드"}
    if not required.issubset(df.columns):
        raise ValueError("keyword.csv에는 '카테고리', '키워드' 두 열이 필요합니다.")

    df = df[["카테고리", "키워드"]].dropna().copy()
    df["카테고리"] = df["카테고리"].astype(str).str.strip()
    df["키워드"] = df["키워드"].astype(str).str.strip()
    return df


def blank_png() -> bytes:
    img = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def snapshot_as_pil(image_bytes: bytes | None) -> Image.Image:
    if not image_bytes:
        image_bytes = blank_png()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def clean_one_word(text: str) -> str:
    """모델 응답을 화면에 표시하기 좋은 한 단어 형태로 정리한다."""
    if not text:
        return "응답없음"

    value = text.strip().splitlines()[0]
    value = value.replace("**", "").replace("`", "")
    value = value.strip(" \t\n\r\"'“”‘’.,!?()[]{}:;")

    # 모델이 실수로 짧은 문장을 반환한 경우 첫 토큰만 사용
    if " " in value:
        value = value.split()[0]

    # 흔한 종결 표현 제거
    value = re.sub(r"(입니다|이에요|예요|같아요|입니다\.)$", "", value).strip()
    return value or "응답없음"


def normalize_word(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(text)).lower()


def ask_gemini(category: str, image_bytes: bytes | None) -> str:
    if "GEMINI_API_KEY" not in st.secrets:
        raise RuntimeError(
            "Streamlit Secrets에 GEMINI_API_KEY가 없습니다. "
            "앱 설정의 Secrets에 API 키를 등록해 주세요."
        )

    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    image = snapshot_as_pil(image_bytes)

    prompt = f"""
너는 초등학생용 AI 캐치마인드 게임의 정답 추론 AI야.

카테고리: {category}

규칙:
1. 제공된 그림의 색상보다 전체적인 형태, 윤곽, 배치와 특징을 우선해서 판단한다.
2. 초등학생이 태블릿으로 짧은 시간 안에 그린 단순한 그림임을 고려한다.
3. 반드시 '{category}' 카테고리에 속한다고 생각되는 대상을 추론한다.
4. 답변은 반드시 한국어 명사 한 단어만 출력한다.
5. 설명, 이유, 조사, 문장, 기호, 따옴표를 절대 붙이지 않는다.

출력 예시: 사과
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[image, prompt],
    )
    return clean_one_word(response.text)


def start_new_round():
    st.session_state.round_started_at = time.time()
    st.session_state.snapshot_bytes = None
    st.session_state.snapshot_json = None
    st.session_state.submitting = False
    st.session_state.timeout_handled = False
    st.session_state.submit_reason = None
    st.session_state.ai_error = None


def finish_round(ai_guess: str):
    answer = st.session_state.questions[st.session_state.round_index]
    image_bytes = st.session_state.snapshot_bytes or blank_png()

    st.session_state.results.append(
        {
            "round": st.session_state.round_index + 1,
            "image": image_bytes,
            "ai": ai_guess,
            "answer": answer,
            "correct": normalize_word(ai_guess) == normalize_word(answer),
        }
    )

    if st.session_state.round_index + 1 >= len(st.session_state.questions):
        st.session_state.stage = "results"
        st.session_state.submitting = False
    else:
        st.session_state.round_index += 1
        start_new_round()

    st.rerun()


def reset_all():
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value
    st.rerun()


# -----------------------------
# 1초 단위 타이머
# -----------------------------
@st.fragment(run_every="1s")
def render_timer():
    if st.session_state.stage != "game" or st.session_state.submitting:
        return

    started = st.session_state.round_started_at
    if started is None:
        return

    elapsed = time.time() - started
    remaining = max(0, math.ceil(ROUND_SECONDS - elapsed))

    st.progress(remaining / ROUND_SECONDS)
    st.markdown(f"### ⏱️ 남은 시간: **{remaining}초**")

    if remaining <= 0 and not st.session_state.timeout_handled:
        st.session_state.timeout_handled = True
        st.session_state.submitting = True
        st.session_state.submit_reason = "timeout"
        st.rerun()


# -----------------------------
# 화면 1: 시작 화면
# -----------------------------
def render_start():
    st.title("🎨 AI 캐치마인드")
    st.write("카테고리를 고르고 그림을 그리면 Gemini AI가 정답을 맞혀요!")

    st.markdown("### 1. 카테고리를 선택하세요")
    category = st.selectbox(
        "카테고리",
        CATEGORIES,
        index=0,
        label_visibility="collapsed",
    )

    st.info("게임은 총 5문제이며, 문제당 제한시간은 60초입니다.")

    if st.button("게임 시작 🚀", use_container_width=True, type="primary"):
        try:
            df = load_keywords()
        except Exception as e:
            st.error(str(e))
            return

        candidates = (
            df.loc[df["카테고리"] == category, "키워드"]
            .drop_duplicates()
            .tolist()
        )

        if len(candidates) < QUESTION_COUNT:
            st.error(
                f"'{category}' 카테고리의 키워드가 {QUESTION_COUNT}개 이상 필요합니다. "
                f"현재 {len(candidates)}개입니다."
            )
            return

        st.session_state.category = category
        st.session_state.questions = random.sample(candidates, QUESTION_COUNT)
        st.session_state.round_index = 0
        st.session_state.results = []
        st.session_state.stage = "game"
        start_new_round()
        st.rerun()


# -----------------------------
# 화면 2: 게임 화면
# -----------------------------
def render_game():
    idx = st.session_state.round_index
    answer = st.session_state.questions[idx]

    st.title("🎨 AI 캐치마인드")
    st.caption(f"카테고리: {st.session_state.category}")

    col1, col2 = st.columns(2)
    col1.metric("문제", f"{idx + 1} / {QUESTION_COUNT}")
    col2.metric("카테고리", st.session_state.category)

    st.markdown(
        f'<div class="keyword-box">제시어: {answer}</div>',
        unsafe_allow_html=True,
    )
    st.caption("제시어를 글자로 쓰지 말고 그림으로 표현해 보세요.")

    # AI 제출 상태: 마지막 그림을 읽기 전용 캔버스로 고정
    if st.session_state.submitting:
        if st.session_state.submit_reason == "timeout":
            st.warning("⏰ 60초가 지나 그림판이 잠겼습니다. 마지막 그림으로 AI가 추론합니다.")

        st_canvas(
            stroke_width=8,
            stroke_color="#111111",
            background_color="#FFFFFF",
            height=CANVAS_HEIGHT,
            width=CANVAS_WIDTH,
            drawing_mode="freedraw",
            initial_drawing=st.session_state.snapshot_json,
            update_streamlit=False,
            return_image_data=False,
            disabled=True,
            key=f"locked_canvas_{idx}",
        )

        st.info("🤔 AI가 생각 중입니다")

        if st.session_state.ai_error:
            st.error(f"AI 호출 중 오류가 발생했습니다: {st.session_state.ai_error}")
            if st.button("AI 다시 호출", use_container_width=True, type="primary"):
                st.session_state.ai_error = None
                st.rerun()
            return

        try:
            with st.spinner("Gemini가 그림의 형태를 분석하고 있어요..."):
                guess = ask_gemini(
                    st.session_state.category,
                    st.session_state.snapshot_bytes,
                )
        except Exception as e:
            st.session_state.ai_error = str(e)
            st.rerun()
            return

        finish_round(guess)
        return

    # 그림을 그리는 상태
    canvas_result = st_canvas(
        stroke_width=8,
        stroke_color="#111111",
        background_color="#FFFFFF",
        height=CANVAS_HEIGHT,
        width=CANVAS_WIDTH,
        drawing_mode="freedraw",
        update_streamlit=True,
        return_image_data=True,
        disabled=False,
        key=f"canvas_{idx}",
    )

    # 매 스트로크가 끝날 때마다 마지막 스냅샷을 보관
    if canvas_result.image_bytes:
        st.session_state.snapshot_bytes = canvas_result.image_bytes
    if canvas_result.json_data is not None:
        st.session_state.snapshot_json = canvas_result.json_data

    render_timer()

    if st.button("제출하기 🤖", use_container_width=True, type="primary"):
        st.session_state.submitting = True
        st.session_state.timeout_handled = True
        st.session_state.submit_reason = "manual"
        st.session_state.ai_error = None
        st.rerun()


# -----------------------------
# 화면 3: 결과 화면
# -----------------------------
def render_results():
    st.title("🏁 게임 결과")

    score = sum(1 for item in st.session_state.results if item["correct"])
    st.metric("AI가 맞힌 문제", f"{score} / {QUESTION_COUNT}")
    st.progress(score / QUESTION_COUNT)

    for item in st.session_state.results:
        status = "✅ 정답" if item["correct"] else "❌ 오답"
        with st.container(border=True):
            st.subheader(f"{item['round']}라운드 · {status}")
            st.image(item["image"], caption="사용자가 그린 그림", use_container_width=True)

            c1, c2 = st.columns(2)
            c1.markdown(f"**AI 응답**  \n{item['ai']}")
            c2.markdown(f"**정답**  \n{item['answer']}")

    if st.button("처음부터 다시 하기", use_container_width=True, type="primary"):
        reset_all()


# -----------------------------
# 앱 실행
# -----------------------------
if st.session_state.stage == "start":
    render_start()
elif st.session_state.stage == "game":
    render_game()
else:
    render_results()
