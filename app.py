import html
import io
import random
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from google import genai
from google.genai import errors, types
from streamlit_drawable_canvas import st_canvas


# ============================================================
# 1. 기본 설정
# ============================================================

st.set_page_config(
    page_title="AI 캐치마인드",
    page_icon="🎨",
    layout="wide",
)

CATEGORIES = [
    "동물",
    "과일",
    "채소",
    "사물",
    "교통수단",
]

MODEL_NAME = "gemini-2.5-flash"

TIME_LIMIT = 60
MAX_PASSES = 2

CANVAS_WIDTH = 720
CANVAS_HEIGHT = 480

# 최초 호출 실패 후
# 2초 → 4초 → 8초 뒤 재시도
RETRY_DELAYS = [2, 4, 8]

# 자동 재시도할 HTTP 오류
RETRYABLE_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}


# ============================================================
# 2. 디자인
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 980px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    .main-title {
        text-align: center;
        font-size: 2.25rem;
        font-weight: 900;
        margin-bottom: 0.6rem;
    }

    .keyword-box {
        background: #f1f3f5;
        border-radius: 18px;
        padding: 18px 20px;
        margin: 8px 0 12px 0;

        text-align: center;
        font-size: 2rem;
        font-weight: 900;
    }

    .timer-box {
        text-align: center;
        font-size: 1.3rem;
        font-weight: 900;
        margin: 8px 0;
    }

    .result-card {
        border: 2px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 16px;
        background: white;
    }

    .correct-card {
        border-color: #22c55e;
        background: #f0fdf4;
    }

    .wrong-card {
        border-color: #ef4444;
        background: #fef2f2;
    }

    .result-status {
        font-size: 1.55rem;
        font-weight: 900;
        margin-bottom: 12px;
    }

    .result-label {
        font-size: 1.15rem;
        font-weight: 800;
        color: #6b7280;
        margin-top: 10px;
    }

    .result-answer {
        font-size: 2rem;
        font-weight: 900;
        line-height: 1.4;
    }

    div.stButton > button {
        min-height: 52px;
        font-size: 1.05rem;
        font-weight: 800;
        border-radius: 14px;
    }

    @media (max-width: 800px) {

        .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }

        .main-title {
            font-size: 1.8rem;
        }

        .keyword-box {
            font-size: 1.65rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. Session State 초기화
# ============================================================

def init_state():

    defaults = {

        "page": "start",

        "category": None,
        "question_count": 5,

        # 실제 문제 + 패스용 문제
        "question_queue": [],

        # 현재 제시어 위치
        "queue_index": 0,

        # 실제 채점 완료 문제 수
        "answered_count": 0,

        "passes_used": 0,

        "results": [],

        # 현재 문제 시작 시각
        "round_start": None,

        # 캔버스 새로 생성용 번호
        "round_token": 0,

        # 현재 그림 마지막 스냅샷
        "last_snapshot": None,

        # 제출 순간 확정한 그림
        "pending_snapshot": None,

        "pending_timed_out": False,

        # AI 처리 상태
        "processing": False,

        # communication / configuration
        "submission_error": None,

        # 시간 종료 여부
        "timeout_triggered": False,

        # 그림 도구
        "stroke_color": "#111111",
        "stroke_width": 8,

        # 다음 문제에서 직전 결과 표시
        "flash_message": None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
# 4. keyword.csv 불러오기
# ============================================================

@st.cache_data
def load_keywords():

    path = Path("keyword.csv")

    if not path.exists():

        raise FileNotFoundError(
            "keyword.csv 파일을 찾을 수 없습니다."
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    required_columns = {
        "카테고리",
        "키워드",
    }

    if not required_columns.issubset(df.columns):

        raise ValueError(
            "keyword.csv에는 "
            "'카테고리', '키워드' 열이 필요합니다."
        )

    df = df[
        ["카테고리", "키워드"]
    ].dropna().copy()

    df["카테고리"] = (
        df["카테고리"]
        .astype(str)
        .str.strip()
    )

    df["키워드"] = (
        df["키워드"]
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["카테고리"] != "")
        &
        (df["키워드"] != "")
    ]

    df = df.drop_duplicates(
        subset=[
            "카테고리",
            "키워드",
        ]
    )

    return df


# ============================================================
# 5. 이미지 처리
# ============================================================

def blank_png():

    image = Image.new(
        "RGB",
        (
            CANVAS_WIDTH,
            CANVAS_HEIGHT,
        ),
        "white",
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def get_canvas_png(canvas_result):

    if canvas_result is None:
        return None

    # --------------------------------------------------------
    # drawable-canvas 0.13.x에서는 image_bytes 제공
    # --------------------------------------------------------

    try:

        image_bytes = canvas_result.image_bytes

        if image_bytes:

            return bytes(image_bytes)

    except Exception:
        pass

    # --------------------------------------------------------
    # image_data fallback
    # --------------------------------------------------------

    try:

        data = canvas_result.image_data

        if data is None:
            return None

        image = Image.fromarray(
            data.astype("uint8"),
            mode="RGBA",
        )

        background = Image.new(
            "RGB",
            image.size,
            "white",
        )

        background.paste(
            image,
            mask=image.getchannel("A"),
        )

        buffer = io.BytesIO()

        background.save(
            buffer,
            format="PNG",
        )

        return buffer.getvalue()

    except Exception as exc:

        print(
            "[Canvas 이미지 변환 오류] "
            f"type={type(exc).__name__}, "
            f"message={str(exc)}"
        )

        return None


def show_locked_canvas(snapshot):

    snapshot = snapshot or blank_png()

    try:

        background = Image.open(
            io.BytesIO(snapshot)
        ).convert("RGB")

    except Exception:

        background = Image.new(
            "RGB",
            (
                CANVAS_WIDTH,
                CANVAS_HEIGHT,
            ),
            "white",
        )

    st_canvas(

        stroke_width=(
            st.session_state.stroke_width
        ),

        stroke_color=(
            st.session_state.stroke_color
        ),

        background_image=background,

        background_image_fit="stretch",

        update_streamlit=False,

        height=CANVAS_HEIGHT,

        width=CANVAS_WIDTH,

        drawing_mode="freedraw",

        # 읽기 전용
        disabled=True,

        key=(
            f"locked_canvas_"
            f"{st.session_state.round_token}"
        ),
    )


# ============================================================
# 6. AI 응답 처리
# ============================================================

def normalize_word(text):

    return re.sub(
        r"[\s\W_]+",
        "",
        str(text),
        flags=re.UNICODE,
    ).lower()


def clean_ai_answer(text):

    if not text:
        return ""

    cleaned = str(text).strip()

    # 혹시
    # "정답: 물고기"
    # 형태로 응답해도 처리
    cleaned = re.sub(
        r"^(정답|답|추측)"
        r"\s*[:：\-]?\s*",
        "",
        cleaned,
    )

    # 첫 단어만 추출
    match = re.search(
        r"[가-힣A-Za-z0-9]+",
        cleaned,
    )

    if match:
        return match.group(0)

    return ""


# ============================================================
# 7. Gemini 연결
# ============================================================

@st.cache_resource
def get_gemini_client():

    if "GEMINI_API_KEY" not in st.secrets:

        raise RuntimeError(
            "Streamlit Secrets에 "
            "GEMINI_API_KEY가 없습니다."
        )

    api_key = str(
        st.secrets["GEMINI_API_KEY"]
    ).strip()

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY가 비어 있습니다."
        )

    # 앱에서 직접 재시도하므로
    # SDK 기본 클라이언트만 사용
    return genai.Client(
        api_key=api_key,
    )


# ============================================================
# 8. Gemini 그림 분석
# ============================================================

def ask_gemini(
    category,
    image_bytes,
):

    prompt = f"""
너는 초등학생용 'AI 캐치마인드' 게임의 그림 맞히기 AI다.

현재 카테고리: {category}

사용자는 초등학생이며,
제시어를 글자로 쓰지 않고 그림으로 표현했다.

그림을 보고 다음 규칙에 따라 정답을 추론하라.

규칙:
1. 세부 묘사보다 전체 윤곽과 형태를 가장 중요하게 본다.
2. 주요 부품의 모양과 위치를 중요하게 본다.
3. 초등학생의 단순하고 서툰 그림임을 고려한다.
4. 색상은 보조적인 단서로만 활용한다.
5. 반드시 '{category}' 카테고리에 속하는 대상만 답한다.
6. 설명이나 이유를 절대 출력하지 않는다.
7. '정답은', '제가 보기에는' 같은 문장을 쓰지 않는다.
8. 반드시 한 단어만 출력한다.

올바른 출력 예:
사과
원숭이
연필

잘못된 출력 예:
정답은 사과입니다.
사과 같아요.
제가 보기에는 원숭이입니다.
"""

    # --------------------------------------------------------
    # Gemini Client 확인
    # --------------------------------------------------------

    try:

        client = get_gemini_client()

    except Exception as exc:

        print(
            "[Gemini 설정 오류] "
            f"type={type(exc).__name__}, "
            f"message={str(exc)}"
        )

        return None, "configuration"

    # --------------------------------------------------------
    # 이미지 Part 생성
    # --------------------------------------------------------

    try:

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png",
        )

    except Exception as exc:

        print(
            "[Gemini 이미지 변환 오류] "
            f"type={type(exc).__name__}, "
            f"message={str(exc)}"
        )

        return None, "communication"

    # --------------------------------------------------------
    # 최초 요청 1회
    # 실패하면 2초 / 4초 / 8초 후 재시도
    # --------------------------------------------------------

    total_attempts = (
        len(RETRY_DELAYS) + 1
    )

    for attempt in range(total_attempts):

        try:

            print(
                "[Gemini 호출] "
                f"{attempt + 1}/{total_attempts}번째 시도"
            )

            response = (
                client.models.generate_content(

                    model=MODEL_NAME,

                    contents=[
                        prompt,
                        image_part,
                    ],

                    config=(
                        types.GenerateContentConfig(
                            temperature=0.1,
                            max_output_tokens=30,
                        )
                    ),
                )
            )

            # ------------------------------------------------
            # 원본 응답 확인
            # ------------------------------------------------

            raw_text = response.text

            print(
                "[Gemini 원본 응답] "
                f"{repr(raw_text)}"
            )

            answer = clean_ai_answer(
                raw_text
            )

            if answer:

                print(
                    "[Gemini 최종 답변] "
                    f"{answer}"
                )

                return answer, None

            print(
                "[Gemini 응답 오류] "
                "응답에서 한 단어를 추출하지 못했습니다."
            )

            return None, "communication"

        # ====================================================
        # Gemini API 오류
        # ====================================================

        except errors.APIError as exc:

            code = getattr(
                exc,
                "code",
                None,
            )

            print(
                "[Gemini API 오류] "
                f"code={code}, "
                f"type={type(exc).__name__}, "
                f"message={str(exc)}"
            )

            # ------------------------------------------------
            # 408 / 429 / 500 / 502 / 503 / 504
            # ------------------------------------------------

            if code in RETRYABLE_CODES:

                if attempt < len(RETRY_DELAYS):

                    wait_time = (
                        RETRY_DELAYS[attempt]
                    )

                    print(
                        "[Gemini 자동 재시도] "
                        f"{wait_time}초 후 재호출"
                    )

                    time.sleep(
                        wait_time
                    )

                    continue

                print(
                    "[Gemini 최종 통신 실패] "
                    "자동 재시도를 모두 사용했습니다."
                )

                return None, "communication"

            # ------------------------------------------------
            # 400 / 401 / 403 등
            # ------------------------------------------------

            print(
                "[Gemini 요청/설정 오류] "
                f"HTTP code={code}"
            )

            return None, "configuration"

        # ====================================================
        # 네트워크 오류
        # ====================================================

        except (
            ConnectionError,
            TimeoutError,
        ) as exc:

            print(
                "[Gemini 네트워크 오류] "
                f"type={type(exc).__name__}, "
                f"message={str(exc)}"
            )

            if attempt < len(RETRY_DELAYS):

                wait_time = (
                    RETRY_DELAYS[attempt]
                )

                print(
                    "[Gemini 자동 재시도] "
                    f"{wait_time}초 후 재호출"
                )

                time.sleep(
                    wait_time
                )

                continue

            return None, "communication"

        # ====================================================
        # 기타 예상하지 못한 오류
        # ====================================================

        except Exception as exc:

            print(
                "[Gemini 기타 오류] "
                f"type={type(exc).__name__}, "
                f"message={str(exc)}"
            )

            if attempt < len(RETRY_DELAYS):

                wait_time = (
                    RETRY_DELAYS[attempt]
                )

                print(
                    "[Gemini 자동 재시도] "
                    f"{wait_time}초 후 재호출"
                )

                time.sleep(
                    wait_time
                )

                continue

            print(
                "[Gemini 기타 오류 최종 실패]"
            )

            return None, "communication"

    return None, "communication"


# ============================================================
# 9. 라운드 초기화
# ============================================================

def reset_round():

    st.session_state.round_start = (
        time.time()
    )

    st.session_state.round_token += 1

    st.session_state.last_snapshot = None

    st.session_state.pending_snapshot = None

    st.session_state.pending_timed_out = False

    st.session_state.processing = False

    st.session_state.submission_error = None

    st.session_state.timeout_triggered = False


# ============================================================
# 10. 게임 시작
# ============================================================

def start_game(
    category,
    question_count,
    keywords,
):

    needed = (
        question_count
        +
        MAX_PASSES
    )

    selected_keywords = random.sample(
        list(keywords),
        needed,
    )

    st.session_state.page = "game"

    st.session_state.category = category

    st.session_state.question_count = (
        question_count
    )

    st.session_state.question_queue = (
        selected_keywords
    )

    st.session_state.queue_index = 0

    st.session_state.answered_count = 0

    st.session_state.passes_used = 0

    st.session_state.results = []

    st.session_state.flash_message = None

    reset_round()

    st.rerun()


# ============================================================
# 11. 다음 문제 / 결과 화면
# ============================================================

def go_next_or_result():

    if (
        st.session_state.answered_count
        >=
        st.session_state.question_count
    ):

        st.session_state.page = "result"

        st.session_state.processing = False

        st.session_state.submission_error = None

        st.rerun()

    reset_round()

    st.rerun()


# ============================================================
# 12. 채점
# ============================================================

def grade_and_save(
    ai_answer,
    snapshot,
    timed_out,
):

    correct_answer = (
        st.session_state.question_queue[
            st.session_state.queue_index
        ]
    )

    is_correct = (
        normalize_word(ai_answer)
        ==
        normalize_word(correct_answer)
    )

    round_number = (
        st.session_state.answered_count
        +
        1
    )

    st.session_state.results.append(
        {
            "round": round_number,
            "image": snapshot,
            "ai_answer": ai_answer,
            "answer": correct_answer,
            "correct": is_correct,
            "timed_out": timed_out,
        }
    )

    st.session_state.answered_count += 1

    st.session_state.queue_index += 1

    if is_correct:

        st.session_state.flash_message = (
            f"✅ 정답! AI도 '{ai_answer}'라고 생각했어요."
        )

    else:

        st.session_state.flash_message = (
            f"❌ 오답! "
            f"AI의 답은 '{ai_answer}', "
            f"정답은 '{correct_answer}'입니다."
        )

    go_next_or_result()


# ============================================================
# 13. 제출 준비
# ============================================================

def queue_submission(
    snapshot,
    timed_out=False,
):

    st.session_state.pending_snapshot = (
        snapshot
        or
        blank_png()
    )

    st.session_state.pending_timed_out = (
        timed_out
    )

    # 캔버스 잠금
    st.session_state.processing = True

    st.session_state.submission_error = None

    st.rerun()


# ============================================================
# 14. AI 실제 호출
# ============================================================

def process_pending_submission():

    snapshot = (
        st.session_state.pending_snapshot
        or
        blank_png()
    )

    ai_answer, error_type = ask_gemini(
        st.session_state.category,
        snapshot,
    )

    # --------------------------------------------------------
    # 호출 실패
    # --------------------------------------------------------

    if ai_answer is None:

        st.session_state.processing = False

        st.session_state.submission_error = (
            error_type
        )

        st.rerun()

    # --------------------------------------------------------
    # 성공 → 즉시 채점
    # --------------------------------------------------------

    grade_and_save(
        ai_answer,
        snapshot,
        st.session_state.pending_timed_out,
    )


# ============================================================
# 15. AI 다시 호출
# ============================================================

def retry_submission():

    st.session_state.processing = True

    st.session_state.submission_error = None

    st.rerun()


# ============================================================
# 16. 패스
# ============================================================

def pass_question():

    if (
        st.session_state.passes_used
        >=
        MAX_PASSES
    ):
        return

    st.session_state.passes_used += 1

    # 현재 문제만 건너뛴다.
    st.session_state.queue_index += 1

    # answered_count는 증가하지 않는다.
    # 즉 패스는 실제 문항 수에 포함되지 않는다.

    remaining = (
        MAX_PASSES
        -
        st.session_state.passes_used
    )

    st.session_state.flash_message = (
        f"⏭️ 패스했습니다. "
        f"남은 패스 {remaining}회"
    )

    reset_round()

    st.rerun()


# ============================================================
# 17. 그림 지우기
# ============================================================

def clear_canvas():

    # 캔버스만 새로 생성
    # 시간은 초기화하지 않는다.
    st.session_state.round_token += 1

    st.session_state.last_snapshot = None

    st.rerun()


# ============================================================
# 18. 전체 게임 초기화
# ============================================================

def restart_game():

    for key in list(
        st.session_state.keys()
    ):

        del st.session_state[key]

    st.rerun()


# ============================================================
# 19. 시작 화면
# ============================================================

def show_start_page():

    st.markdown(
        """
        <div class="main-title">
            🎨 AI 캐치마인드
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        "카테고리와 문제 수를 선택한 뒤 "
        "게임을 시작해 보세요."
    )

    # --------------------------------------------------------
    # CSV 로딩
    # --------------------------------------------------------

    try:

        df = load_keywords()

    except Exception as exc:

        st.error(
            str(exc)
        )

        st.stop()

    # --------------------------------------------------------
    # 카테고리 선택
    # --------------------------------------------------------

    category = st.selectbox(
        "카테고리",
        CATEGORIES,
    )

    keywords = (
        df.loc[
            df["카테고리"] == category,
            "키워드",
        ]
        .drop_duplicates()
        .tolist()
    )

    # 패스용 키워드 2개를 확보해야 함
    max_questions = (
        len(keywords)
        -
        MAX_PASSES
    )

    if max_questions < 1:

        st.error(
            f"'{category}' 카테고리는 "
            f"최소 {MAX_PASSES + 1}개의 "
            "키워드가 필요합니다."
        )

        st.stop()

    default_count = min(
        5,
        max_questions,
    )

    # --------------------------------------------------------
    # 문제 수
    # --------------------------------------------------------

    question_count = st.number_input(
        "문제 수",
        min_value=1,
        max_value=max_questions,
        value=default_count,
        step=1,
    )

    st.caption(
        f"'{category}' 키워드 {len(keywords)}개 · "
        f"게임에서는 "
        f"{int(question_count)}문항 + "
        f"패스용 {MAX_PASSES}문항을 준비합니다."
    )

    # --------------------------------------------------------
    # 시작
    # --------------------------------------------------------

    if st.button(
        "🚀 게임 시작",
        type="primary",
        use_container_width=True,
    ):

        start_game(
            category,
            int(question_count),
            keywords,
        )


# ============================================================
# 20. 타이머
# ============================================================

@st.fragment(
    run_every="1s"
)
def show_timer():

    if (
        st.session_state.page != "game"
        or
        st.session_state.processing
        or
        st.session_state.submission_error
        or
        st.session_state.round_start is None
    ):
        return

    elapsed = (
        time.time()
        -
        st.session_state.round_start
    )

    remaining = max(
        0,
        TIME_LIMIT - int(elapsed),
    )

    st.markdown(
        f"""
        <div class="timer-box">
            ⏱️ 남은 시간: {remaining}초
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(
        remaining / TIME_LIMIT
    )

    # --------------------------------------------------------
    # 시간 종료
    # --------------------------------------------------------

    if (
        remaining <= 0
        and
        not st.session_state.timeout_triggered
    ):

        st.session_state.timeout_triggered = True

        # 전체 앱을 다시 실행하여
        # 마지막 스냅샷으로 자동 제출
        st.rerun()


# ============================================================
# 21. 제출 / AI 처리 화면
# ============================================================

def show_submission_screen():

    snapshot = (
        st.session_state.pending_snapshot
        or
        st.session_state.last_snapshot
        or
        blank_png()
    )

    # --------------------------------------------------------
    # 마지막 그림 고정 표시
    # --------------------------------------------------------

    show_locked_canvas(
        snapshot
    )

    # --------------------------------------------------------
    # AI 처리 중
    # --------------------------------------------------------

    if st.session_state.processing:

        st.info(
            "🤔 AI가 생각 중입니다"
        )

        with st.spinner(
            "그림을 분석하고 있어요..."
        ):

            process_pending_submission()

        return

    # --------------------------------------------------------
    # 통신 실패
    # --------------------------------------------------------

    if (
        st.session_state.submission_error
        ==
        "communication"
    ):

        st.error(
            "통신에 실패했습니다"
        )

        st.caption(
            "일시적인 네트워크 문제나 "
            "AI 서버 혼잡일 수 있습니다. "
            "자동 재시도 후에도 실패한 상태입니다."
        )

        if st.button(
            "🔄 AI 다시 호출",
            type="primary",
            use_container_width=True,
        ):

            retry_submission()

        return

    # --------------------------------------------------------
    # API KEY / 요청 설정 오류
    # --------------------------------------------------------

    if (
        st.session_state.submission_error
        ==
        "configuration"
    ):

        st.error(
            "AI 설정 오류가 발생했습니다."
        )

        st.caption(
            "Streamlit Secrets의 "
            "GEMINI_API_KEY 또는 "
            "Gemini API 설정을 확인해 주세요."
        )

        if st.button(
            "🔄 다시 시도",
            use_container_width=True,
        ):

            retry_submission()


# ============================================================
# 22. 게임 화면
# ============================================================

def show_game_page():

    # --------------------------------------------------------
    # 게임 종료 확인
    # --------------------------------------------------------

    if (
        st.session_state.answered_count
        >=
        st.session_state.question_count
    ):

        st.session_state.page = "result"

        st.rerun()

    # --------------------------------------------------------
    # 제시어 부족 예외
    # --------------------------------------------------------

    if (
        st.session_state.queue_index
        >=
        len(st.session_state.question_queue)
    ):

        st.error(
            "준비된 제시어가 부족합니다. "
            "게임을 다시 시작해 주세요."
        )

        if st.button(
            "처음으로"
        ):

            restart_game()

        return

    # --------------------------------------------------------
    # 직전 결과 알림
    # --------------------------------------------------------

    if st.session_state.flash_message:

        message = (
            st.session_state.flash_message
        )

        st.session_state.flash_message = None

        st.toast(
            message
        )

    keyword = (
        st.session_state.question_queue[
            st.session_state.queue_index
        ]
    )

    current_round = (
        st.session_state.answered_count
        +
        1
    )

    remaining_passes = (
        MAX_PASSES
        -
        st.session_state.passes_used
    )

    # --------------------------------------------------------
    # 제목
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="main-title">
            🎨 AI 캐치마인드
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # 상태
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "카테고리",
            st.session_state.category,
        )

    with col2:

        st.metric(
            "문제",
            f"{current_round} / "
            f"{st.session_state.question_count}",
        )

    with col3:

        st.metric(
            "남은 패스",
            f"{remaining_passes}회",
        )

    # --------------------------------------------------------
    # 제시어
    # --------------------------------------------------------

    safe_keyword = html.escape(
        str(keyword)
    )

    st.markdown(
        f"""
        <div class="keyword-box">
            제시어: {safe_keyword}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "제시어를 글자로 쓰지 말고 "
        "그림으로 표현해 보세요."
    )

    # ========================================================
    # AI 처리 또는 오류 상태
    # ========================================================

    if (
        st.session_state.processing
        or
        st.session_state.submission_error
    ):

        show_submission_screen()

        return

    # ========================================================
    # 시간 초과
    # ========================================================

    if st.session_state.timeout_triggered:

        st.warning(
            "⏰ 시간이 종료되었습니다. "
            "마지막 그림을 저장하고 "
            "AI가 정답을 추론합니다."
        )

        queue_submission(
            (
                st.session_state.last_snapshot
                or
                blank_png()
            ),
            timed_out=True,
        )

        return

    # ========================================================
    # 타이머
    # ========================================================

    show_timer()

    # ========================================================
    # 색상 팔레트
    # ========================================================

    st.subheader(
        "🎨 색상 팔레트"
    )

    palette = [
        ("⚫ 검정", "#111111"),
        ("🔴 빨강", "#E53935"),
        ("🔵 파랑", "#1E88E5"),
        ("🟢 초록", "#43A047"),
    ]

    palette_columns = st.columns(4)

    for column, (label, color) in zip(
        palette_columns,
        palette,
    ):

        with column:

            if st.button(
                label,
                use_container_width=True,
                key=f"palette_{color}",
            ):

                st.session_state.stroke_color = color

                st.rerun()

    # --------------------------------------------------------
    # 추가 색상 / 굵기
    # --------------------------------------------------------

    tool1, tool2 = st.columns(2)

    with tool1:

        st.color_picker(
            "다른 색 고르기",
            key="stroke_color",
        )

    with tool2:

        st.slider(
            "펜 굵기",
            min_value=3,
            max_value=24,
            key="stroke_width",
        )

    # ========================================================
    # 그림판
    # ========================================================

    canvas_result = st_canvas(

        fill_color="rgba(255, 255, 255, 0)",

        stroke_width=(
            st.session_state.stroke_width
        ),

        stroke_color=(
            st.session_state.stroke_color
        ),

        background_color="#FFFFFF",

        update_streamlit=True,

        height=CANVAS_HEIGHT,

        width=CANVAS_WIDTH,

        drawing_mode="freedraw",

        # 0.13.x에서 이미지 반환 필수
        return_image_data=True,

        # display_toolbar 사용하지 않음
        key=(
            f"canvas_"
            f"{st.session_state.round_token}"
        ),
    )

    # --------------------------------------------------------
    # 최신 그림 스냅샷 저장
    # --------------------------------------------------------

    snapshot = get_canvas_png(
        canvas_result
    )

    if snapshot:

        st.session_state.last_snapshot = (
            snapshot
        )

    # ========================================================
    # 하단 버튼
    # ========================================================

    submit_col, pass_col, clear_col = (
        st.columns(
            [
                1.3,
                1,
                1,
            ]
        )
    )

    with submit_col:

        submit_clicked = st.button(
            "✅ 제출",
            type="primary",
            use_container_width=True,
        )

    with pass_col:

        pass_clicked = st.button(
            f"⏭️ 패스 ({remaining_passes}회)",
            use_container_width=True,
            disabled=(
                remaining_passes <= 0
            ),
        )

    with clear_col:

        clear_clicked = st.button(
            "🧹 그림 지우기",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # 제출
    # --------------------------------------------------------

    if submit_clicked:

        queue_submission(
            (
                st.session_state.last_snapshot
                or
                blank_png()
            ),
            timed_out=False,
        )

    # --------------------------------------------------------
    # 패스
    # --------------------------------------------------------

    if pass_clicked:

        pass_question()

    # --------------------------------------------------------
    # 그림 지우기
    # --------------------------------------------------------

    if clear_clicked:

        clear_canvas()


# ============================================================
# 23. 결과 화면
# ============================================================

def show_result_page():

    st.markdown(
        """
        <div class="main-title">
            🏁 게임 결과
        </div>
        """,
        unsafe_allow_html=True,
    )

    results = (
        st.session_state.results
    )

    score = sum(
        1
        for item in results
        if item["correct"]
    )

    # --------------------------------------------------------
    # 전체 결과
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "총 문제",
            f"{len(results)}문제",
        )

    with col2:

        st.metric(
            "정답",
            f"{score}개",
        )

    with col3:

        st.metric(
            "사용한 패스",
            f"{st.session_state.passes_used}회",
        )

    if results:

        st.progress(
            score / len(results)
        )

        st.markdown(
            f"### 🎉 점수: "
            f"**{score} / {len(results)}**"
        )

    # ========================================================
    # 라운드별 결과
    # ========================================================

    for item in results:

        if item["correct"]:

            status = "✅ 정답"
            card_class = "correct-card"
            status_color = "#15803d"

        else:

            status = "❌ 오답"
            card_class = "wrong-card"
            status_color = "#b91c1c"

        if item["timed_out"]:

            timeout_note = (
                " · ⏰ 시간 초과"
            )

        else:

            timeout_note = ""

        st.markdown("---")

        st.subheader(
            f"{item['round']}라운드"
        )

        left, right = st.columns(
            [1.15, 1]
        )

        # ----------------------------------------------------
        # 사용자 그림
        # ----------------------------------------------------

        with left:

            st.image(
                item["image"],
                caption="사용자가 그린 그림",
                use_container_width=True,
            )

        # ----------------------------------------------------
        # AI 응답 / 정답
        # ----------------------------------------------------

        with right:

            safe_ai_answer = html.escape(
                str(item["ai_answer"])
            )

            safe_answer = html.escape(
                str(item["answer"])
            )

            st.markdown(
                f"""
                <div class="result-card {card_class}">

                    <div
                        class="result-status"
                        style="color:{status_color};"
                    >
                        {status}{timeout_note}
                    </div>

                    <div class="result-label">
                        🤖 AI 응답
                    </div>

                    <div class="result-answer">
                        {safe_ai_answer}
                    </div>

                    <div class="result-label">
                        🎯 정답
                    </div>

                    <div class="result-answer">
                        {safe_answer}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # 다시 시작
    # --------------------------------------------------------

    st.markdown("---")

    if st.button(
        "🔁 다시 게임하기",
        type="primary",
        use_container_width=True,
    ):

        restart_game()


# ============================================================
# 24. 페이지 실행
# ============================================================

if st.session_state.page == "start":

    show_start_page()

elif st.session_state.page == "game":

    show_game_page()

else:

    show_result_page()
