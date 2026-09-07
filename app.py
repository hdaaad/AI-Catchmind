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


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="AI 캐치마인드",
    page_icon="🎨",
    layout="wide",
)

CATEGORIES = ["동물", "과일", "채소", "사물", "교통수단"]

MODEL_NAME = "gemini-2.5-flash"

TIME_LIMIT = 60
MAX_PASSES = 2

CANVAS_WIDTH = 720
CANVAS_HEIGHT = 480

# 최초 호출 실패 후
# 2초 → 4초 → 8초 뒤 재시도
RETRY_DELAYS = [2, 4, 8]

RETRYABLE_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}


# =========================================================
# 디자인
# =========================================================
st.markdown(
    """
    <style>

    .block-container {
        max-width: 980px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    .title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 900;
        margin-bottom: .5rem;
    }

    .keyword {
        background: #f2f4f7;
        border-radius: 18px;
        padding: 16px 20px;
        text-align: center;
        font-size: 2rem;
        font-weight: 900;
        margin: 8px 0 10px;
    }

    .timer {
        text-align: center;
        font-size: 1.25rem;
        font-weight: 900;
        margin: .3rem 0;
    }

    .result-card {
        border: 2px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 14px;
        background: white;
    }

    .correct {
        border-color: #22c55e;
        background: #f0fdf4;
    }

    .wrong {
        border-color: #ef4444;
        background: #fef2f2;
    }

    .big-label {
        font-size: 1.1rem;
        font-weight: 800;
        color: #6b7280;
        margin-top: 8px;
    }

    .big-answer {
        font-size: 2rem;
        font-weight: 900;
        line-height: 1.3;
    }

    div.stButton > button {
        min-height: 52px;
        font-size: 1.05rem;
        font-weight: 800;
        border-radius: 14px;
    }

    @media (max-width: 800px) {

        .block-container {
            padding-left: .6rem;
            padding-right: .6rem;
        }

        .title {
            font-size: 1.8rem;
        }

        .keyword {
            font-size: 1.65rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Session State 초기화
# =========================================================
def init_state():

    defaults = {

        "page": "start",

        "category": None,

        "question_count": 5,

        # 문항수 + 패스용 2문항
        "question_queue": [],

        "queue_index": 0,

        # 실제 채점 완료 문항 수
        "answered_count": 0,

        "passes_used": 0,

        "results": [],

        # 현재 라운드 시작 시간
        "round_start": None,

        # 새로운 캔버스를 생성하기 위한 번호
        "round_token": 0,

        # 현재 캔버스의 마지막 그림
        "last_snapshot": None,

        # 제출 순간 확정된 그림
        "pending_snapshot": None,

        "pending_timed_out": False,

        # AI 처리 상태
        "processing": False,

        # communication / configuration
        "submission_error": None,

        # 시간 종료 여부
        "timeout_triggered": False,

        # 그림 설정
        "stroke_color": "#111111",

        "stroke_width": 8,

        # 직전 문제 결과 메시지
        "flash_message": None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


init_state()


# =========================================================
# keyword.csv 읽기
# =========================================================
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

    required = {
        "카테고리",
        "키워드",
    }

    if not required.issubset(df.columns):

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
        ["카테고리", "키워드"]
    )

    return df


# =========================================================
# 이미지 관련
# =========================================================
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
    """
    drawable-canvas 0.13.0의 image_bytes를 우선 사용.
    불가능하면 image_data를 PNG로 변환.
    """

    if canvas_result is None:

        return None

    # 0.13.0 image_bytes
    try:

        if canvas_result.image_bytes:

            return bytes(
                canvas_result.image_bytes
            )

    except Exception:

        pass

    # fallback
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

    except Exception:

        return None


def show_locked_canvas(snapshot):
    """
    제출 또는 시간초과 이후
    마지막 그림을 읽기 전용 캔버스로 고정 표시.
    """

    snapshot = (
        snapshot
        or
        blank_png()
    )

    background = Image.open(
        io.BytesIO(snapshot)
    ).convert("RGB")

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

        # 0.13.0 지원
        disabled=True,

        key=(
            f"locked_"
            f"{st.session_state.round_token}"
        ),
    )


# =========================================================
# AI 응답 정리
# =========================================================
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

    # 혹시 AI가
    # "정답: 사과"라고 출력해도 정리
    cleaned = re.sub(

        r"^(정답|답|추측)"
        r"\s*[:：\-]?\s*",

        "",

        cleaned,
    )

    # 첫 단어만 사용
    match = re.search(

        r"[가-힣A-Za-z0-9]+",

        cleaned,
    )

    if match:

        return match.group(0)

    return ""


# =========================================================
# Gemini 연결
# =========================================================
@st.cache_resource
def get_gemini_client():

    if "GEMINI_API_KEY" not in st.secrets:

        raise RuntimeError(
            "Streamlit Secrets에 "
            "GEMINI_API_KEY가 없습니다."
        )

    # SDK 자체의 중복 자동 재시도를 방지하고
    # 아래 ask_gemini 함수에서 직접 2→4→8초 재시도
    return genai.Client(

        api_key=(
            st.secrets[
                "GEMINI_API_KEY"
            ]
        ),

        http_options=types.HttpOptions(

            retry_options=(
                types.HttpRetryOptions(
                    attempts=1
                )
            )
        ),
    )


def ask_gemini(
    category,
    image_bytes,
):

    prompt = f"""
너는 초등학생용 AI 캐치마인드 게임의 그림 맞히기 AI다.

카테고리: {category}

초등학생이 제시어를 글자로 쓰지 않고 그림으로 표현했다.

다음 규칙을 반드시 지켜라.

1. 세부 묘사보다 전체 윤곽, 주요 형태,
   부품의 위치와 배치를 가장 중요하게 본다.

2. 색상은 보조 단서로만 사용한다.

3. 반드시 '{category}' 카테고리에 속하는
   대상 하나만 추론한다.

4. 설명, 이유, 문장, 조사, 따옴표,
   마침표를 출력하지 않는다.

5. 최종 응답은 반드시 한 단어만 출력한다.

출력 예시:
사과
원숭이
연필
"""

    try:

        client = get_gemini_client()

    except Exception:

        return None, "configuration"


    image_part = types.Part.from_bytes(

        data=image_bytes,

        mime_type="image/png",
    )


    # 최초 호출
    # +
    # 실패하면 2초 / 4초 / 8초 후 재시도
    for attempt in range(
        len(RETRY_DELAYS) + 1
    ):

        try:

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

                            max_output_tokens=20,

                            # 단순 이미지 분류이므로
                            # Thinking 비활성화
                            thinking_config=(
                                types.ThinkingConfig(
                                    thinking_budget=0
                                )
                            ),
                        )
                    ),
                )
            )

            answer = clean_ai_answer(
                response.text
            )

            if answer:

                return answer, None

            return None, "communication"


        # Gemini API 오류
        except errors.APIError as exc:

            code = getattr(
                exc,
                "code",
                None,
            )

            # 503 / 429 등
            if (
                code in RETRYABLE_CODES
                and
                attempt < len(
                    RETRY_DELAYS
                )
            ):

                time.sleep(
                    RETRY_DELAYS[
                        attempt
                    ]
                )

                continue


            # 자동 재시도 후에도 실패
            if code in RETRYABLE_CODES:

                return (
                    None,
                    "communication",
                )


            # API Key 등 설정 문제
            return (
                None,
                "configuration",
            )


        # 일반 네트워크 오류
        except (
            ConnectionError,
            TimeoutError,
        ):

            if attempt < len(
                RETRY_DELAYS
            ):

                time.sleep(
                    RETRY_DELAYS[
                        attempt
                    ]
                )

                continue

            return (
                None,
                "communication",
            )


        # 기타 예상하지 못한 통신 오류
        except Exception:

            if attempt < len(
                RETRY_DELAYS
            ):

                time.sleep(
                    RETRY_DELAYS[
                        attempt
                    ]
                )

                continue

            return (
                None,
                "communication",
            )


    return (
        None,
        "communication",
    )


# =========================================================
# 라운드 초기화
# =========================================================
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


# =========================================================
# 게임 시작
# =========================================================
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

    st.session_state.page = "game"

    st.session_state.category = category

    st.session_state.question_count = (
        question_count
    )

    # 문항 수 + 패스 2개
    st.session_state.question_queue = (
        random.sample(
            list(keywords),
            needed,
        )
    )

    st.session_state.queue_index = 0

    st.session_state.answered_count = 0

    st.session_state.passes_used = 0

    st.session_state.results = []

    st.session_state.flash_message = None

    reset_round()

    st.rerun()


# =========================================================
# 다음 라운드 또는 결과
# =========================================================
def go_next_or_result():

    if (
        st.session_state.answered_count
        >=
        st.session_state.question_count
    ):

        st.session_state.page = (
            "result"
        )

        st.session_state.processing = (
            False
        )

        st.session_state.submission_error = (
            None
        )

        st.rerun()

    reset_round()

    st.rerun()


# =========================================================
# 채점
# =========================================================
def grade_and_save(
    ai_answer,
    snapshot,
    timed_out,
):

    correct_answer = (
        st.session_state
        .question_queue[
            st.session_state.queue_index
        ]
    )

    is_correct = (

        normalize_word(
            ai_answer
        )

        ==

        normalize_word(
            correct_answer
        )
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
            "✅ 정답입니다!"
        )

    else:

        st.session_state.flash_message = (
            f"❌ 오답입니다. "
            f"정답은 "
            f"'{correct_answer}'입니다."
        )


    go_next_or_result()


# =========================================================
# 제출 준비
# =========================================================
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

    # 캔버스를 잠그고
    # 다음 rerun에서 AI 호출
    st.session_state.processing = True

    st.session_state.submission_error = None

    st.rerun()


# =========================================================
# 실제 AI 호출
# =========================================================
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


    if ai_answer is None:

        st.session_state.processing = (
            False
        )

        st.session_state.submission_error = (
            error_type
        )

        st.rerun()


    grade_and_save(

        ai_answer,

        snapshot,

        st.session_state.pending_timed_out,
    )


# =========================================================
# AI 재호출
# =========================================================
def retry_submission():

    st.session_state.processing = True

    st.session_state.submission_error = None

    st.rerun()


# =========================================================
# 패스
# =========================================================
def pass_question():

    if (
        st.session_state.passes_used
        >=
        MAX_PASSES
    ):

        return


    st.session_state.passes_used += 1

    # 현재 제시어 건너뛰기
    st.session_state.queue_index += 1


    st.session_state.flash_message = (

        "⏭️ 패스했습니다. "
        f"남은 패스 "
        f"{MAX_PASSES - st.session_state.passes_used}회"
    )


    # answered_count는 증가시키지 않음
    # 따라서 패스 문제는 문제 수에 포함되지 않음

    reset_round()

    st.rerun()


# =========================================================
# 그림 지우기
# =========================================================
def clear_canvas():

    # canvas key를 변경하여
    # 새 캔버스를 생성
    st.session_state.round_token += 1

    st.session_state.last_snapshot = None

    # 시간은 초기화하지 않음
    st.rerun()


# =========================================================
# 게임 완전 초기화
# =========================================================
def restart_game():

    for key in list(
        st.session_state.keys()
    ):

        del st.session_state[key]

    st.rerun()


# =========================================================
# 시작 화면
# =========================================================
def show_start_page():

    st.markdown(

        '<div class="title">'
        '🎨 AI 캐치마인드'
        '</div>',

        unsafe_allow_html=True,
    )


    st.write(
        "카테고리를 고르고 문제 수를 선택한 뒤 "
        "게임을 시작하세요."
    )


    try:

        df = load_keywords()

    except Exception as exc:

        st.error(
            str(exc)
        )

        st.stop()


    # 카테고리 선택
    category = st.selectbox(

        "카테고리",

        CATEGORIES,
    )


    keywords = (

        df.loc[
            df["카테고리"]
            ==
            category,

            "키워드"
        ]

        .drop_duplicates()

        .tolist()
    )


    # 패스용 2개 필요
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


    question_count = st.number_input(

        "문항 수",

        min_value=1,

        max_value=max_questions,

        value=default_count,

        step=1,
    )


    st.caption(

        f"현재 {len(keywords)}개 키워드 · "

        f"게임 시작 시 "

        f"{int(question_count) + MAX_PASSES}개"

        "(문항 수 + 패스용 2개)를 "

        "무작위로 준비합니다."
    )


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


# =========================================================
# 타이머
# =========================================================
@st.fragment(
    run_every="1s"
)
def show_timer():

    if (
        st.session_state.page
        !=
        "game"

        or

        st.session_state.processing

        or

        st.session_state.submission_error

        or

        st.session_state.round_start
        is None
    ):

        return


    elapsed = (

        time.time()

        -

        st.session_state.round_start
    )


    remaining = max(

        0,

        TIME_LIMIT
        -
        int(elapsed),
    )


    st.markdown(

        f'<div class="timer">'
        f'⏱️ 남은 시간: '
        f'{remaining}초'
        f'</div>',

        unsafe_allow_html=True,
    )


    st.progress(

        remaining
        /
        TIME_LIMIT
    )


    # 60초 종료
    if (
        remaining <= 0

        and

        not
        st.session_state.timeout_triggered
    ):

        st.session_state.timeout_triggered = (
            True
        )

        # 전체 앱 rerun
        st.rerun()


# =========================================================
# 제출 이후 고정 화면
# =========================================================
def show_submission_screen():

    snapshot = (

        st.session_state.pending_snapshot

        or

        st.session_state.last_snapshot

        or

        blank_png()
    )


    # 마지막 스냅샷을
    # disabled canvas로 고정
    show_locked_canvas(
        snapshot
    )


    # ---------------------------------
    # AI 처리
    # ---------------------------------
    if st.session_state.processing:

        st.info(
            "🤔 AI가 생각 중입니다"
        )

        with st.spinner(
            "그림을 분석하고 있어요..."
        ):

            process_pending_submission()

        return


    # ---------------------------------
    # 통신 실패
    # ---------------------------------
    if (
        st.session_state.submission_error
        ==
        "communication"
    ):

        # 요구사항 그대로 출력
        st.error(
            "통신에 실패했습니다"
        )


        st.caption(

            "503 서버 혼잡 또는 일시적인 "
            "네트워크 문제일 수 있습니다. "
            "자동 재시도 후에도 실패한 상태입니다."
        )


        if st.button(

            "🔄 AI 다시 호출",

            type="primary",

            use_container_width=True,
        ):

            retry_submission()

        return


    # ---------------------------------
    # API 설정 오류
    # ---------------------------------
    if (
        st.session_state.submission_error
        ==
        "configuration"
    ):

        st.error(

            "AI 설정 오류입니다. "
            "Streamlit Secrets의 "
            "GEMINI_API_KEY와 "
            "모델 설정을 확인해 주세요."
        )

        return


# =========================================================
# 게임 화면
# =========================================================
def show_game_page():

    # 게임 종료
    if (
        st.session_state.answered_count
        >=
        st.session_state.question_count
    ):

        st.session_state.page = (
            "result"
        )

        st.rerun()


    # 혹시 큐가 부족한 경우
    if (
        st.session_state.queue_index
        >=
        len(
            st.session_state.question_queue
        )
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


    # 직전 채점 결과
    if st.session_state.flash_message:

        st.toast(
            st.session_state.flash_message
        )

        st.session_state.flash_message = (
            None
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


    # 제목
    st.markdown(

        '<div class="title">'
        '🎨 AI 캐치마인드'
        '</div>',

        unsafe_allow_html=True,
    )


    # 상태 표시
    col1, col2, col3 = st.columns(3)


    col1.metric(

        "카테고리",

        st.session_state.category,
    )


    col2.metric(

        "문제",

        f"{current_round} / "
        f"{st.session_state.question_count}",
    )


    col3.metric(

        "남은 패스",

        f"{remaining_passes}회",
    )


    # 제시어
    st.markdown(

        f'<div class="keyword">'
        f'제시어: {keyword}'
        f'</div>',

        unsafe_allow_html=True,
    )


    st.caption(

        "제시어를 글자로 쓰지 말고 "
        "그림으로 표현해 보세요."
    )


    # =====================================================
    # AI 처리 중 또는 통신 오류
    # =====================================================
    if (
        st.session_state.processing

        or

        st.session_state.submission_error
    ):

        show_submission_screen()

        return


    # =====================================================
    # 시간 초과
    # =====================================================
    if st.session_state.timeout_triggered:

        st.warning(

            "⏰ 시간이 종료되었습니다. "
            "마지막 그림으로 "
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


    # 타이머
    show_timer()


    # =====================================================
    # 색상 팔레트
    # =====================================================
    st.subheader(
        "🎨 색상 팔레트"
    )


    palette = [

        (
            "⚫ 검정",
            "#111111",
        ),

        (
            "🔴 빨강",
            "#E53935",
        ),

        (
            "🔵 파랑",
            "#1E88E5",
        ),

        (
            "🟢 초록",
            "#43A047",
        ),
    ]


    palette_columns = st.columns(4)


    for (
        column,
        (
            label,
            color,
        )
    ) in zip(
        palette_columns,
        palette,
    ):

        with column:

            if st.button(

                label,

                use_container_width=True,

                key=(
                    f"palette_"
                    f"{color}"
                ),
            ):

                st.session_state.stroke_color = (
                    color
                )

                st.rerun()


    # 다른 색 / 굵기
    tool1, tool2 = st.columns(2)


    with tool1:

        st.color_picker(

            "다른 색 고르기",

            key="stroke_color",
        )


    with tool2:

        st.slider(

            "펜 굵기",

            3,

            24,

            key="stroke_width",
        )


    # =====================================================
    # 실제 그림판
    # =====================================================
    canvas_result = st_canvas(

        fill_color=(
            "rgba(255, 255, 255, 0)"
        ),

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

        # ★ 0.13.0에서 반드시 필요
        return_image_data=True,

        # display_toolbar 사용하지 않음
        key=(
            f"canvas_"
            f"{st.session_state.round_token}"
        ),
    )


    # 캔버스 최신 이미지 저장
    snapshot = get_canvas_png(
        canvas_result
    )


    if snapshot:

        st.session_state.last_snapshot = (
            snapshot
        )


    # =====================================================
    # 하단 버튼
    # =====================================================
    button1, button2, button3 = (
        st.columns(
            [
                1.3,
                1,
                1,
            ]
        )
    )


    with button1:

        submit_clicked = st.button(

            "✅ 제출",

            type="primary",

            use_container_width=True,
        )


    with button2:

        pass_clicked = st.button(

            f"⏭️ 패스 "
            f"({remaining_passes}회)",

            use_container_width=True,

            disabled=(
                remaining_passes <= 0
            ),
        )


    with button3:

        clear_clicked = st.button(

            "🧹 그림 지우기",

            use_container_width=True,
        )


    # 제출
    if submit_clicked:

        queue_submission(

            (
                st.session_state.last_snapshot

                or

                blank_png()
            ),

            timed_out=False,
        )


    # 패스
    if pass_clicked:

        pass_question()


    # 지우기
    if clear_clicked:

        clear_canvas()


# =========================================================
# 결과 화면
# =========================================================
def show_result_page():

    st.markdown(

        '<div class="title">'
        '🏁 게임 결과'
        '</div>',

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


    # 전체 점수
    col1, col2, col3 = st.columns(3)


    col1.metric(

        "총 문제",

        f"{len(results)}문제",
    )


    col2.metric(

        "정답",

        f"{score}개",
    )


    col3.metric(

        "사용한 패스",

        f"{st.session_state.passes_used}회",
    )


    if results:

        st.progress(

            score
            /
            len(results)
        )


        st.markdown(

            f"### 🎉 점수: "
            f"**{score} / {len(results)}**"
        )


    # =====================================================
    # 라운드별 결과
    # =====================================================
    for item in results:

        if item["correct"]:

            status = "✅ 정답"

            css_class = "correct"

            status_color = "#15803d"

        else:

            status = "❌ 오답"

            css_class = "wrong"

            status_color = "#b91c1c"


        if item["timed_out"]:

            timeout_note = (
                " · ⏰ 시간 초과"
            )

        else:

            timeout_note = ""


        st.markdown("---")


        st.subheader(

            f"{item['round']}라운드 "
            f"· {status}"
        )


        left, right = st.columns(
            [
                1.15,
                1,
            ]
        )


        # 그림
        with left:

            st.image(

                item["image"],

                caption=(
                    "사용자가 그린 그림"
                ),

                use_container_width=True,
            )


        # 결과
        with right:

            st.markdown(

                f"""
                <div class="result-card {css_class}">

                    <div style="
                        font-size:1.45rem;
                        font-weight:900;
                        color:{status_color};
                    ">
                        {status}{timeout_note}
                    </div>

                    <div class="big-label">
                        🤖 AI 응답
                    </div>

                    <div class="big-answer">
                        {item["ai_answer"]}
                    </div>

                    <div class="big-label">
                        🎯 정답
                    </div>

                    <div class="big-answer">
                        {item["answer"]}
                    </div>

                </div>
                """,

                unsafe_allow_html=True,
            )


    st.markdown("---")


    if st.button(

        "🔁 다시 게임하기",

        type="primary",

        use_container_width=True,
    ):

        restart_game()


# =========================================================
# 페이지 이동
# =========================================================
if st.session_state.page == "start":

    show_start_page()


elif st.session_state.page == "game":

    show_game_page()


else:

    show_result_page()
