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


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="AI 캐치마인드",
    page_icon="🎨",
    layout="wide",
)

CATEGORIES = ["동물", "과일", "채소", "사물", "교통수단"]

TIME_LIMIT = 60
MAX_PASSES = 2

CANVAS_WIDTH = 720
CANVAS_HEIGHT = 480

MODEL_NAME = "gemini-2.5-flash"

# 503, 429 등의 일시적 오류 발생 시
# 최초 호출 후 2초 → 4초 → 8초 간격으로 재시도
RETRY_DELAYS = [2, 4, 8]
RETRYABLE_CODES = {429, 500, 502, 503, 504}


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

    .game-title {
        text-align: center;
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: .5rem;
    }

    .keyword-box {
        background: #f2f4f7;
        border-radius: 18px;
        padding: 16px 20px;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        margin: 8px 0 14px 0;
    }

    .timer-box {
        font-size: 1.25rem;
        font-weight: 800;
        text-align: center;
        padding: 6px;
    }

    .result-card {
        border-radius: 18px;
        padding: 18px 20px;
        margin: 8px 0 16px 0;
        border: 2px solid #e5e7eb;
        background: white;
    }

    .result-correct {
        border-color: #22c55e;
        background: #f0fdf4;
    }

    .result-wrong {
        border-color: #ef4444;
        background: #fef2f2;
    }

    .result-label {
        font-size: 1.05rem;
        font-weight: 700;
        color: #6b7280;
        margin-top: 8px;
    }

    .result-answer {
        font-size: 1.8rem;
        font-weight: 900;
        line-height: 1.35;
    }

    div.stButton > button {
        min-height: 52px;
        font-size: 1.05rem;
        font-weight: 800;
        border-radius: 14px;
    }

    @media (max-width: 800px) {

        .block-container {
            padding-left: .7rem;
            padding-right: .7rem;
        }

        .game-title {
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


# =========================================================
# Session State 초기화
# =========================================================
def init_state():

    defaults = {
        "page": "start",

        "category": None,
        "question_count": 5,

        # 문항수 + 패스 대비 2개
        "question_queue": [],

        # 실제 keyword 목록에서 현재 위치
        "queue_index": 0,

        # 실제 채점 완료 문제 수
        "answered_count": 0,

        "passes_used": 0,

        "results": [],

        # 현재 문제 시작 시간
        "round_start": None,

        # 캔버스를 새로 만들 때 사용하는 번호
        "round_token": 0,

        # 현재 그림 최신 스냅샷
        "last_snapshot": None,

        # 제출 순간 확정된 그림
        "pending_snapshot": None,

        # 시간초과 제출인지
        "pending_timed_out": False,

        # AI 호출 중인지
        "processing": False,

        # AI 통신/설정 오류
        "submission_error": None,

        # 자동 제출 여부
        "auto_submit": False,

        # 그림 도구
        "stroke_color": "#111111",
        "stroke_width": 8,

        # 다음 문제에서 잠시 보여줄 메시지
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
        encoding="utf-8-sig"
    )

    required = {
        "카테고리",
        "키워드"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "keyword.csv에는 '카테고리', '키워드' 열이 필요합니다."
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
        subset=["카테고리", "키워드"]
    )

    return df


# =========================================================
# 이미지 처리
# =========================================================
def blank_png():

    image = Image.new(
        "RGB",
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        "white"
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


def canvas_to_png(image_data):

    if image_data is None:
        return None

    image = Image.fromarray(
        image_data.astype("uint8"),
        mode="RGBA"
    )

    # 투명 영역을 흰색으로 변환
    background = Image.new(
        "RGB",
        image.size,
        "white"
    )

    background.paste(
        image,
        mask=image.getchannel("A")
    )

    buffer = io.BytesIO()

    background.save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


# =========================================================
# 단어 비교
# =========================================================
def normalize_word(text):

    return re.sub(
        r"[\s\W_]+",
        "",
        str(text),
        flags=re.UNICODE
    ).lower()


def clean_ai_answer(text):

    if not text:
        return ""

    cleaned = str(text).strip()

    # Gemini가 실수로 "정답: 사과"처럼 응답해도 정리
    cleaned = re.sub(
        r"^(정답|답|추측)\s*[:：\-]?\s*",
        "",
        cleaned
    )

    # 첫 번째 단어만 추출
    match = re.search(
        r"[가-힣A-Za-z0-9]+",
        cleaned
    )

    if match:
        return match.group(0)

    return ""


# =========================================================
# Gemini API
# =========================================================
@st.cache_resource
def get_gemini_client():

    if "GEMINI_API_KEY" not in st.secrets:

        raise RuntimeError(
            "Streamlit Secrets에 "
            "GEMINI_API_KEY가 설정되어 있지 않습니다."
        )

    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )


def ask_gemini(category, image_bytes):

    prompt = f"""
너는 초등학생용 'AI 캐치마인드' 게임의 그림 맞히기 AI다.

카테고리: {category}

아래 이미지는 초등학생이 제시어를 글자로 쓰지 않고
그림으로 표현한 것이다.

규칙:

1. 초등학생의 단순한 그림이므로 세부 묘사보다
   전체 윤곽, 주요 형태, 부품의 위치와 배치를
   가장 중요하게 본다.

2. 색상은 보조 단서로만 사용한다.

3. 반드시 '{category}' 카테고리에 속하는
   대상 하나만 추론한다.

4. 설명, 이유, 조사, 문장, 목록, 따옴표,
   마침표를 절대 출력하지 않는다.

5. 최종 응답은 반드시 한 단어만 출력한다.

출력 예시:

사과

원숭이

연필
"""

    client = get_gemini_client()

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/png"
    )

    # 최초 호출 + 최대 3회 재시도
    for attempt in range(
        len(RETRY_DELAYS) + 1
    ):

        try:

            response = client.models.generate_content(

                model=MODEL_NAME,

                contents=[
                    image_part,
                    prompt
                ],

                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=20,
                ),
            )

            answer = clean_ai_answer(
                response.text
            )

            if not answer:
                raise RuntimeError(
                    "AI 응답에서 단어를 추출하지 못했습니다."
                )

            return answer, None


        # Gemini API가 반환한 오류
        except errors.APIError as exc:

            code = getattr(
                exc,
                "code",
                None
            )

            # 503, 429 등 일시적 오류
            if (
                code in RETRYABLE_CODES
                and
                attempt < len(RETRY_DELAYS)
            ):

                time.sleep(
                    RETRY_DELAYS[attempt]
                )

                continue


            # 자동 재시도를 모두 사용함
            if code in RETRYABLE_CODES:

                return None, "communication"


            # API KEY, 권한 등의 설정 오류
            return None, "configuration"


        # 일반 네트워크 오류
        except (
            ConnectionError,
            TimeoutError
        ):

            if attempt < len(RETRY_DELAYS):

                time.sleep(
                    RETRY_DELAYS[attempt]
                )

                continue

            return None, "communication"


        # 그 외 예상하지 못한 통신 오류
        except Exception:

            if attempt < len(RETRY_DELAYS):

                time.sleep(
                    RETRY_DELAYS[attempt]
                )

                continue

            return None, "communication"


    return None, "communication"


# =========================================================
# 게임 상태 관리
# =========================================================
def reset_for_new_round():

    st.session_state.round_start = (
        time.time()
    )

    st.session_state.round_token += 1

    st.session_state.last_snapshot = None

    st.session_state.pending_snapshot = None

    st.session_state.pending_timed_out = False

    st.session_state.processing = False

    st.session_state.submission_error = None

    st.session_state.auto_submit = False


def start_game(
    category,
    question_count,
    category_keywords
):

    # 패스 2회를 대비해서
    # 문제 수 + 2개의 제시어 준비
    needed = (
        question_count
        +
        MAX_PASSES
    )

    selected = random.sample(
        list(category_keywords),
        needed
    )

    st.session_state.page = "game"

    st.session_state.category = category

    st.session_state.question_count = (
        question_count
    )

    st.session_state.question_queue = (
        selected
    )

    st.session_state.queue_index = 0

    st.session_state.answered_count = 0

    st.session_state.passes_used = 0

    st.session_state.results = []

    st.session_state.flash_message = None

    reset_for_new_round()


def finish_or_next_round():

    # 선택한 문제 수만큼 실제로 채점했으면 종료
    if (
        st.session_state.answered_count
        >=
        st.session_state.question_count
    ):

        st.session_state.page = "result"

        st.session_state.processing = False

        st.session_state.submission_error = None

        st.session_state.auto_submit = False

        st.rerun()

    # 다음 문제 시작
    reset_for_new_round()

    st.rerun()


# =========================================================
# 채점
# =========================================================
def grade_and_save(
    ai_answer,
    snapshot,
    timed_out
):

    correct_answer = (
        st.session_state.question_queue[
            st.session_state.queue_index
        ]
    )

    round_number = (
        st.session_state.answered_count
        +
        1
    )

    is_correct = (
        normalize_word(ai_answer)
        ==
        normalize_word(correct_answer)
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
            f"❌ 아쉬워요! "
            f"정답은 '{correct_answer}'이었어요."
        )

    finish_or_next_round()


# =========================================================
# 제출
# =========================================================
def queue_submission(
    snapshot,
    timed_out=False
):

    # 제출 순간의 그림을 확정
    st.session_state.pending_snapshot = (
        snapshot
    )

    st.session_state.pending_timed_out = (
        timed_out
    )

    # 다음 화면부터 캔버스를 잠금
    st.session_state.processing = True

    st.session_state.submission_error = None

    st.rerun()


def run_pending_ai():

    snapshot = (
        st.session_state.pending_snapshot
        or
        blank_png()
    )

    timed_out = (
        st.session_state.pending_timed_out
    )

    with st.spinner(
        "🤔 AI가 생각 중입니다"
    ):

        ai_answer, error_type = (
            ask_gemini(
                st.session_state.category,
                snapshot
            )
        )

    st.session_state.processing = False

    # AI 호출 실패
    if ai_answer is None:

        st.session_state.submission_error = (
            error_type
        )

        st.rerun()

    # 성공하면 바로 채점
    grade_and_save(
        ai_answer,
        snapshot,
        timed_out
    )


def retry_pending_submission():

    # 기존 스냅샷 그대로 다시 호출
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

    # 현재 키워드만 건너뜀
    st.session_state.queue_index += 1

    st.session_state.flash_message = (
        "⏭️ 패스했습니다. "
        f"남은 패스 "
        f"{MAX_PASSES - st.session_state.passes_used}회"
    )

    # answered_count는 증가시키지 않음
    # → 패스 문제는 문제 수에 포함되지 않음

    reset_for_new_round()

    st.rerun()


# =========================================================
# 그림 초기화
# =========================================================
def clear_canvas():

    # 문제와 타이머는 유지하고
    # 새로운 빈 캔버스만 생성
    st.session_state.round_token += 1

    st.session_state.last_snapshot = None

    st.rerun()


# =========================================================
# 게임 초기화
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
        '<div class="game-title">'
        '🎨 AI 캐치마인드'
        '</div>',
        unsafe_allow_html=True
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


    # -------------------------
    # 카테고리
    # -------------------------
    category = st.selectbox(
        "카테고리",
        CATEGORIES
    )


    keywords = (

        df.loc[
            df["카테고리"] == category,
            "키워드"
        ]

        .drop_duplicates()

        .tolist()
    )


    # 패스용 2개를 제외한 만큼까지
    # 실제 문항으로 설정 가능
    max_questions = (
        len(keywords)
        -
        MAX_PASSES
    )


    if max_questions < 1:

        st.error(

            f"'{category}' 카테고리는 최소 "
            f"{MAX_PASSES + 1}개의 키워드가 필요합니다. "
            "keyword.csv를 확인해 주세요."
        )

        st.stop()


    default_count = min(
        5,
        max_questions
    )


    # -------------------------
    # 문항 수
    # -------------------------
    question_count = st.number_input(

        "문항 수",

        min_value=1,

        max_value=max_questions,

        value=default_count,

        step=1,

        help=(
            "패스 2회에 대비하여 "
            "실제로는 선택한 문항 수보다 "
            "2개의 키워드를 더 준비합니다."
        )
    )


    st.caption(

        f"현재 '{category}' 키워드 "
        f"{len(keywords)}개 · "

        f"게임 시작 시 "
        f"{int(question_count) + MAX_PASSES}개를 "
        "무작위로 준비합니다."
    )


    # -------------------------
    # 게임 시작
    # -------------------------
    if st.button(
        "🚀 게임 시작",
        type="primary",
        use_container_width=True
    ):

        start_game(
            category,
            int(question_count),
            keywords
        )


# =========================================================
# 제출 후 고정 화면
# =========================================================
def show_frozen_submission():

    snapshot = (

        st.session_state.pending_snapshot

        or

        st.session_state.last_snapshot

        or

        blank_png()
    )


    # 기존 캔버스 대신
    # 마지막 그림을 그대로 표시
    st.subheader(
        "제출한 그림"
    )

    st.image(
        snapshot,
        width=CANVAS_WIDTH
    )


    # -------------------------
    # AI 처리 중
    # -------------------------
    if st.session_state.processing:

        st.info(
            "🤔 AI가 생각 중입니다"
        )

        return


    # -------------------------
    # 네트워크 / 503 오류
    # -------------------------
    if (
        st.session_state.submission_error
        ==
        "communication"
    ):

        st.error(
            "통신에 실패했습니다"
        )

        st.caption(

            "503 서버 혼잡이나 일시적인 네트워크 오류는 "
            "자동으로 2초 → 4초 → 8초 간격으로 "
            "재시도한 뒤 이 화면이 표시됩니다."
        )


        if st.button(
            "🔄 AI 다시 호출",
            type="primary",
            use_container_width=True
        ):

            retry_pending_submission()

        return


    # -------------------------
    # API KEY 등의 설정 오류
    # -------------------------
    if (
        st.session_state.submission_error
        ==
        "configuration"
    ):

        st.error(
            "AI 설정을 확인해 주세요. "
            "API 키 또는 모델 설정에 "
            "문제가 있을 수 있습니다."
        )

        if st.button(
            "🔄 다시 시도",
            use_container_width=True
        ):

            retry_pending_submission()


# =========================================================
# 게임 화면
# =========================================================
def show_game_page():

    # -------------------------
    # 종료 확인
    # -------------------------
    if (
        st.session_state.answered_count
        >=
        st.session_state.question_count
    ):

        st.session_state.page = "result"

        st.rerun()


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


    # -------------------------
    # 직전 문제 결과 알림
    # -------------------------
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


    # -------------------------
    # 제목
    # -------------------------
    st.markdown(

        '<div class="game-title">'
        '🎨 AI 캐치마인드'
        '</div>',

        unsafe_allow_html=True
    )


    # -------------------------
    # 상태 표시
    # -------------------------
    top1, top2, top3 = st.columns(
        [1.2, 1, 1]
    )


    with top1:

        st.metric(
            "카테고리",
            st.session_state.category
        )


    with top2:

        st.metric(
            "문제",
            f"{current_round} / "
            f"{st.session_state.question_count}"
        )


    with top3:

        st.metric(
            "남은 패스",
            f"{remaining_passes}회"
        )


    # -------------------------
    # 제시어
    # -------------------------
    st.markdown(

        f'<div class="keyword-box">'
        f'제시어: {html.escape(keyword)}'
        f'</div>',

        unsafe_allow_html=True
    )


    st.caption(
        "제시어를 글자로 쓰지 말고 "
        "그림으로 표현해 보세요."
    )


    # =====================================================
    # AI 제출 처리 중
    # =====================================================
    if st.session_state.processing:

        show_frozen_submission()

        run_pending_ai()

        return


    # =====================================================
    # AI 호출 실패
    # =====================================================
    if st.session_state.submission_error:

        show_frozen_submission()

        return


    # =====================================================
    # 타이머
    # =====================================================
    @st.fragment(
        run_every="1s"
    )
    def countdown():

        elapsed = (
            time.time()
            -
            st.session_state.round_start
        )


        remaining = max(
            0,
            TIME_LIMIT - int(elapsed)
        )


        st.markdown(

            f'<div class="timer-box">'
            f'⏱️ 남은 시간: {remaining}초'
            f'</div>',

            unsafe_allow_html=True
        )


        st.progress(
            remaining / TIME_LIMIT
        )


        # 60초 종료
        if (
            remaining <= 0
            and
            not st.session_state.auto_submit
        ):

            st.session_state.auto_submit = True

            st.rerun()


    countdown()


    # =====================================================
    # 시간초과 자동 제출
    # =====================================================
    if st.session_state.auto_submit:

        st.session_state.auto_submit = False


        snapshot = (

            st.session_state.last_snapshot

            or

            blank_png()
        )


        st.warning(
            "⏰ 시간이 종료되었습니다. "
            "마지막 그림을 저장하고 "
            "AI가 정답을 추론합니다."
        )


        queue_submission(
            snapshot,
            timed_out=True
        )

        return


    # =====================================================
    # 색상 팔레트
    # =====================================================
    st.subheader(
        "🎨 색상 팔레트"
    )


    color_cols = st.columns(4)


    color_options = [

        (
            "⚫ 검정",
            "#111111"
        ),

        (
            "🔴 빨강",
            "#E53935"
        ),

        (
            "🔵 파랑",
            "#1E88E5"
        ),

        (
            "🟢 초록",
            "#43A047"
        ),
    ]


    for (
        col,
        (label, color)
    ) in zip(
        color_cols,
        color_options
    ):

        with col:

            if st.button(
                label,
                use_container_width=True,
                key=f"color_{color}"
            ):

                st.session_state.stroke_color = (
                    color
                )

                st.rerun()


    # -------------------------
    # 추가 색상 / 굵기
    # -------------------------
    tool1, tool2 = st.columns(
        [1, 1]
    )


    with tool1:

        st.color_picker(
            "다른 색 고르기",
            key="stroke_color"
        )


    with tool2:

        st.slider(
            "펜 굵기",
            3,
            24,
            key="stroke_width"
        )


    st.caption(
        f"현재 펜 색상: "
        f"{st.session_state.stroke_color}"
    )


    # =====================================================
    # 그림판
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

        display_toolbar=True,

        key=(
            f"canvas_"
            f"{st.session_state.round_token}"
        ),
    )


    # -------------------------
    # 최신 그림 스냅샷 저장
    # -------------------------
    if (
        canvas_result.image_data
        is not None
    ):

        snapshot = canvas_to_png(
            canvas_result.image_data
        )

        if snapshot:

            st.session_state.last_snapshot = (
                snapshot
            )


    # =====================================================
    # 버튼
    # 팔레트/캔버스 아래 한 줄
    # =====================================================
    action1, action2, action3 = (
        st.columns(
            [1.3, 1, 1]
        )
    )


    with action1:

        submit_clicked = st.button(

            "✅ 제출",

            type="primary",

            use_container_width=True
        )


    with action2:

        pass_clicked = st.button(

            f"⏭️ 패스 "
            f"({remaining_passes}회)",

            use_container_width=True,

            disabled=(
                remaining_passes <= 0
            )
        )


    with action3:

        clear_clicked = st.button(

            "🧹 그림 지우기",

            use_container_width=True
        )


    # -------------------------
    # 그림 지우기
    # -------------------------
    if clear_clicked:

        clear_canvas()


    # -------------------------
    # 패스
    # -------------------------
    if pass_clicked:

        pass_question()


    # -------------------------
    # 제출
    # -------------------------
    if submit_clicked:

        snapshot = (

            st.session_state.last_snapshot

            or

            blank_png()
        )


        queue_submission(
            snapshot,
            timed_out=False
        )


# =========================================================
# 결과 화면
# =========================================================
def show_result_page():

    st.markdown(

        '<div class="game-title">'
        '🏁 게임 결과'
        '</div>',

        unsafe_allow_html=True
    )


    results = (
        st.session_state.results
    )


    score = sum(
        1
        for item in results
        if item["correct"]
    )


    # =====================================================
    # 전체 결과
    # =====================================================
    col1, col2, col3 = (
        st.columns(3)
    )


    col1.metric(
        "총 문제",
        f"{len(results)}문제"
    )


    col2.metric(
        "정답",
        f"{score}개"
    )


    col3.metric(
        "사용한 패스",
        f"{st.session_state.passes_used}회"
    )


    if results:

        st.progress(
            score / len(results)
        )


        st.markdown(
            f"### 🎉 점수: "
            f"**{score} / {len(results)}**"
        )


    # =====================================================
    # 라운드별 결과
    # =====================================================
    for item in results:

        is_correct = (
            item["correct"]
        )


        if is_correct:

            css_class = (
                "result-correct"
            )

            status = (
                "✅ 정답"
            )

            status_color = (
                "#15803d"
            )

        else:

            css_class = (
                "result-wrong"
            )

            status = (
                "❌ 오답"
            )

            status_color = (
                "#b91c1c"
            )


        st.markdown("---")


        st.subheader(
            f"{item['round']}라운드 "
            f"· {status}"
        )


        left, right = st.columns(
            [1.15, 1]
        )


        # -------------------------
        # 사용자가 그린 그림
        # -------------------------
        with left:

            st.image(

                item["image"],

                caption=(
                    "사용자가 그린 그림"
                ),

                use_container_width=True
            )


        # -------------------------
        # AI 답 / 정답
        # -------------------------
        with right:

            if item["timed_out"]:

                timeout_text = (
                    " · ⏰ 시간 초과 자동 제출"
                )

            else:

                timeout_text = ""


            st.markdown(

                f"""
                <div class="result-card {css_class}">

                    <div style="
                        font-size:1.45rem;
                        font-weight:900;
                        color:{status_color};
                    ">
                        {status}{timeout_text}
                    </div>

                    <div class="result-label">
                        🤖 AI 응답
                    </div>

                    <div class="result-answer">
                        {html.escape(item["ai_answer"])}
                    </div>

                    <div class="result-label">
                        🎯 정답
                    </div>

                    <div class="result-answer">
                        {html.escape(item["answer"])}
                    </div>

                </div>
                """,

                unsafe_allow_html=True
            )


    st.markdown("---")


    # =====================================================
    # 다시 시작
    # =====================================================
    if st.button(

        "🔁 다시 게임하기",

        type="primary",

        use_container_width=True
    ):

        restart_game()


# =========================================================
# 페이지 라우팅
# =========================================================
if st.session_state.page == "start":

    show_start_page()


elif st.session_state.page == "game":

    show_game_page()


else:

    show_result_page()
