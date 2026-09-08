import random
import re
import time
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from streamlit_drawable_canvas import st_canvas


# ============================================================
# 기본 설정
# ============================================================

st.set_page_config(
    page_title="AI 캐치마인드",
    page_icon="🎨",
    layout="centered",
)

ROUND_LIMIT_SEC = 60
TOTAL_ROUNDS = 5
MAX_PASSES = 2

CANVAS_WIDTH = 620
CANVAS_HEIGHT = 360

DISPLAY_IMG_WIDTH = 420
THUMB_IMG_WIDTH = 170

# ------------------------------------------------------------
# Gemini 모델
# ------------------------------------------------------------
# 이미지 입력을 지원하는 저비용·경량 안정 모델
# ------------------------------------------------------------

GEMINI_MODEL = "gemini-3.1-flash-lite"


CATEGORIES = {
    "동물": "🐶",
    "과일": "🍎",
    "채소": "🥕",
    "사물": "📎",
    "교통수단": "🚗",
}


# ============================================================
# 디자인
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 760px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    .game-info {
        text-align: center;
        font-size: 15px;
        color: #666666;
        margin-bottom: 4px;
    }

    .keyword-box {
        text-align: center;
        font-size: 30px;
        font-weight: 900;
        background: #f3f4f6;
        border-radius: 16px;
        padding: 14px;
        margin: 8px 0 12px 0;
    }

    .timer-box {
        text-align: center;
        font-size: 17px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .correct-box {
        background: #ecfdf5;
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
    }

    .wrong-box {
        background: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
    }

    .answer-label {
        font-size: 16px;
        color: #666666;
        font-weight: 700;
    }

    .answer-text {
        font-size: 29px;
        font-weight: 900;
        margin-bottom: 8px;
    }

    div.stButton > button {
        min-height: 50px;
        font-size: 16px;
        font-weight: 700;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session State 초기화
# ============================================================

DEFAULTS = {
    "page": "start",

    "category": None,

    "keyword_pool": [],
    "keyword_pointer": 0,
    "current_item": None,

    "round_idx": 0,
    "passes_used": 0,
    "rounds": [],

    "round_start_time": None,

    "draw_seq": 0,

    "last_canvas_data": None,

    # 중복 제출 방지
    "submitted_seq": None,

    # 시간 종료 중복 처리 방지
    "timeout_seq": None,

    # 제출 대기 정보
    "pending_image_bytes": None,
    "pending_item": None,
    "pending_timed_out": False,

    # AI 오류
    "ai_error_type": None,
    "ai_error_message": None,
    "ai_retry_seconds": None,

    # 그림 도구
    "stroke_color": "#000000",
    "stroke_width": 8,
}


for key, value in DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# keyword.csv 읽기
# ============================================================

@st.cache_data
def load_keywords():

    df = pd.read_csv(
        "keyword.csv",
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

    # 유사정답은 없어도 됨
    if "유사정답" not in df.columns:
        df["유사정답"] = ""

    df = df[
        [
            "카테고리",
            "키워드",
            "유사정답",
        ]
    ].copy()

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

    return df


# ============================================================
# Gemini Client
# ============================================================

def get_gemini_client():

    try:

        # Secret 존재 여부 확인
        if "GEMINI_API_KEY" not in st.secrets:

            return None, (
                "Streamlit Secrets에서 "
                "GEMINI_API_KEY를 찾을 수 없습니다."
            )

        api_key = str(
            st.secrets["GEMINI_API_KEY"]
        ).strip()

        if not api_key:

            return None, (
                "GEMINI_API_KEY 값이 비어 있습니다."
            )

        client = genai.Client(
            api_key=api_key
        )

        return client, None

    except Exception as exc:

        error_message = (
            f"{type(exc).__name__}: {str(exc)}"
        )

        print(
            "[Gemini Client 생성 오류]",
            error_message,
        )

        return None, error_message


# ============================================================
# 이미지 처리
# ============================================================
def show_api_diagnostic():
    st.markdown("### 🔧 Gemini API 진단")

    try:
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("❌ GEMINI_API_KEY가 Streamlit Secrets에 없습니다.")
            return

        api_key = str(
            st.secrets["GEMINI_API_KEY"]
        ).strip()

        if not api_key:
            st.error("❌ GEMINI_API_KEY 값이 비어 있습니다.")
            return

        st.success(
            f"✅ API 키를 읽었습니다. "
            f"길이: {len(api_key)}자"
        )

        client = genai.Client(
            api_key=api_key
        )

        st.success("✅ Gemini 클라이언트 생성 성공")

        try:
            models = list(
                client.models.list()
            )

            st.success(
                f"✅ Gemini API 연결 성공 "
                f"({len(models)}개 모델 확인)"
            )

            flash_models = [
                m.name
                for m in models
                if "flash" in m.name.lower()
            ]

            st.write(
                "사용 가능한 Flash 계열 모델:"
            )

            for model in flash_models[:20]:
                st.code(model)

        except Exception as exc:
            st.error("❌ Gemini API 연결 실패")
            st.code(
                f"{type(exc).__name__}: {str(exc)}"
            )

    except Exception as exc:
        st.error("❌ Secrets 읽기 실패")
        st.code(
            f"{type(exc).__name__}: {str(exc)}"
        )
def blank_canvas_image():

    return Image.new(
        "RGB",
        (
            CANVAS_WIDTH,
            CANVAS_HEIGHT,
        ),
        "white",
    )


def canvas_array_to_pil(
    image_data,
):

    rgba = Image.fromarray(
        image_data.astype("uint8"),
        mode="RGBA",
    )

    white_background = Image.new(
        "RGB",
        rgba.size,
        "white",
    )

    white_background.paste(
        rgba,
        mask=rgba.getchannel("A"),
    )

    return white_background


def pil_to_png_bytes(
    image,
):

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def optimize_image_for_ai(
    image,
):

    """
    화면에는 원본 그림을 유지하고
    Gemini로 보낼 때만 작게 줄입니다.
    """

    optimized = image.copy()

    optimized.thumbnail(
        (384, 384)
    )

    if optimized.mode != "RGB":
        optimized = optimized.convert(
            "RGB"
        )

    return optimized


# ============================================================
# AI 답변 정리
# ============================================================

def clean_ai_answer(
    text,
):

    if not text:
        return ""

    text = str(text).strip()

    # 정답: 사과 → 사과
    text = re.sub(
        r"^(정답|답|추측)"
        r"\s*[:：\-]?\s*",
        "",
        text,
    )

    match = re.search(
        r"[가-힣A-Za-z0-9]+",
        text,
    )

    if match:
        return match.group(0)

    return ""


# ============================================================
# 429 재시도 시간 추출
# ============================================================

def extract_retry_seconds(
    message,
):

    patterns = [
        r"retry in ([0-9.]+)s",
        r"retryDelay['\"]?\s*:\s*['\"]?([0-9]+)s",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            message,
            re.IGNORECASE,
        )

        if match:

            try:

                return int(
                    float(
                        match.group(1)
                    )
                ) + 1

            except Exception:
                pass

    return 60


# ============================================================
# Gemini 호출
# ============================================================

def ask_ai_guess(
    image: Image.Image,
    category: str,
):

    client, client_error = (
        get_gemini_client()
    )

    # --------------------------------------------------------
    # API 키 설정 오류
    # --------------------------------------------------------

    if client is None:

        return {
            "success": False,
            "type": "configuration",
            "message": client_error,
        }

    # --------------------------------------------------------
    # AI 전송용 이미지 축소
    # --------------------------------------------------------

    ai_image = optimize_image_for_ai(
        image
    )

    # --------------------------------------------------------
    # 짧은 프롬프트
    # --------------------------------------------------------

    prompt = (
        f"초등학생이 그린 그림이다. "
        f"카테고리는 '{category}'이다. "
        f"형태와 윤곽을 중심으로 무엇인지 추론하라. "
        f"반드시 '{category}'에 속하는 대상 하나를 "
        f"설명 없이 한 단어로만 답하라."
    )

    # 최초 호출 + 503일 때만 한 번 재시도
    max_attempts = 2

    for attempt in range(
        max_attempts
    ):

        try:

            print(
                "[Gemini 호출]",
                f"model={GEMINI_MODEL}",
                f"attempt={attempt + 1}",
            )

            # 공식 SDK는 PIL.Image 입력을 직접 지원
            response = (
                client.models.generate_content(

                    model=GEMINI_MODEL,

                    contents=[
                        prompt,
                        ai_image,
                    ],

                    config=(
                        types.GenerateContentConfig(
                            temperature=0.1,
                            max_output_tokens=16,
                        )
                    ),
                )
            )

            raw_text = (
                response.text
                or
                ""
            )

            print(
                "[Gemini 원본 응답]",
                repr(raw_text),
            )

            answer = clean_ai_answer(
                raw_text
            )

            if not answer:

                return {
                    "success": False,
                    "type": "empty",
                    "message": (
                        "AI가 답을 생성하지 못했습니다."
                    ),
                }

            return {
                "success": True,
                "answer": answer,
            }

        # ====================================================
        # Gemini API 오류
        # ====================================================

        except Exception as exc:

            message = str(exc)

            print(
                "[Gemini API 오류]",
                type(exc).__name__,
                message,
            )

            # ------------------------------------------------
            # 429 - 무료 API 한도 초과
            #
            # 자동 재호출하지 않음
            # ------------------------------------------------

            if (
                "429" in message
                or
                "RESOURCE_EXHAUSTED" in message
            ):

                retry_seconds = (
                    extract_retry_seconds(
                        message
                    )
                )

                return {
                    "success": False,
                    "type": "quota",
                    "retry_seconds": retry_seconds,
                    "message": (
                        "Gemini 무료 사용량 제한에 "
                        "도달했습니다."
                    ),
                }

            # ------------------------------------------------
            # 503 - 서버 혼잡
            # ------------------------------------------------

            if (
                "503" in message
                or
                "UNAVAILABLE" in message
            ):

                if attempt == 0:

                    time.sleep(2)

                    continue

                return {
                    "success": False,
                    "type": "server",
                    "message": (
                        "Gemini 서버가 현재 혼잡합니다. "
                        "잠시 후 다시 시도해 주세요."
                    ),
                }

            # ------------------------------------------------
            # 404 - 모델 문제
            # ------------------------------------------------

            if (
                "404" in message
                or
                "NOT_FOUND" in message
            ):

                return {
                    "success": False,
                    "type": "model",
                    "message": (
                        f"{GEMINI_MODEL} 모델을 "
                        f"현재 사용할 수 없습니다.\n\n"
                        f"실제 오류: {message}"
                    ),
                }

            # ------------------------------------------------
            # 401
            # ------------------------------------------------

            if "401" in message:

                return {
                    "success": False,
                    "type": "configuration",
                    "message": (
                        "Gemini API 키 인증에 실패했습니다.\n\n"
                        f"실제 오류: {message}"
                    ),
                }

            # ------------------------------------------------
            # 403
            # ------------------------------------------------

            if "403" in message:

                return {
                    "success": False,
                    "type": "configuration",
                    "message": (
                        "Gemini API 사용 권한이 없습니다.\n\n"
                        f"실제 오류: {message}"
                    ),
                }

            # ------------------------------------------------
            # 400
            # ------------------------------------------------

            if "400" in message:

                return {
                    "success": False,
                    "type": "request",
                    "message": (
                        "Gemini 요청 형식 오류입니다.\n\n"
                        f"실제 오류: {message}"
                    ),
                }

            # ------------------------------------------------
            # 그 외
            # ------------------------------------------------

            return {
                "success": False,
                "type": "unknown",
                "message": (
                    "AI 호출 중 예상하지 못한 "
                    "오류가 발생했습니다.\n\n"
                    f"실제 오류: {message}"
                ),
            }

    return {
        "success": False,
        "type": "unknown",
        "message": (
            "AI 호출에 실패했습니다."
        ),
    }


# ============================================================
# 정답 판정
# ============================================================

def normalize_answer(
    text,
):

    return re.sub(
        r"\s+",
        "",
        str(text).strip(),
    )


def is_correct(
    ai_answer,
    item,
):

    valid_answers = {
        normalize_answer(
            item["키워드"]
        )
    }

    similar = item.get(
        "유사정답",
        "",
    )

    if (
        isinstance(similar, str)
        and
        similar.strip()
    ):

        for answer in similar.split(
            "|"
        ):

            answer = answer.strip()

            if answer:

                valid_answers.add(
                    normalize_answer(
                        answer
                    )
                )

    return (
        normalize_answer(
            ai_answer
        )
        in
        valid_answers
    )


# ============================================================
# 다음 제시어
# ============================================================

def draw_next_keyword():

    pool = (
        st.session_state.keyword_pool
    )

    pointer = (
        st.session_state.keyword_pointer
    )

    previous_word = None

    if st.session_state.current_item:

        previous_word = (
            st.session_state.current_item[
                "키워드"
            ]
        )

    # --------------------------------------------------------
    # 제시어를 모두 사용했으면 다시 섞음
    # --------------------------------------------------------

    if pointer >= len(pool):

        pool = pool[:]

        random.shuffle(
            pool
        )

        if (
            previous_word
            and
            len(pool) > 1
            and
            pool[0]["키워드"]
            ==
            previous_word
        ):

            pool[0], pool[1] = (
                pool[1],
                pool[0],
            )

        pointer = 0

        st.session_state.keyword_pool = (
            pool
        )

    # --------------------------------------------------------
    # 현재 문제 설정
    # --------------------------------------------------------

    st.session_state.current_item = (
        pool[pointer]
    )

    st.session_state.keyword_pointer = (
        pointer + 1
    )

    st.session_state.round_start_time = (
        time.time()
    )

    st.session_state.last_canvas_data = (
        None
    )

    st.session_state.draw_seq += 1

    st.session_state.submitted_seq = (
        None
    )

    st.session_state.timeout_seq = (
        None
    )


# ============================================================
# 게임 시작
# ============================================================

def start_new_game(
    category,
):

    df = load_keywords()

    subset = (
        df.loc[
            df["카테고리"]
            ==
            category
        ]
        .to_dict(
            "records"
        )
    )

    if not subset:

        st.error(
            "해당 카테고리에 "
            "제시어가 없습니다."
        )

        return

    random.shuffle(
        subset
    )

    st.session_state.category = (
        category
    )

    st.session_state.keyword_pool = (
        subset
    )

    st.session_state.keyword_pointer = 0

    st.session_state.current_item = (
        None
    )

    st.session_state.round_idx = 0

    st.session_state.passes_used = 0

    st.session_state.rounds = []

    st.session_state.pending_image_bytes = (
        None
    )

    st.session_state.pending_item = (
        None
    )

    st.session_state.ai_error_type = (
        None
    )

    st.session_state.ai_error_message = (
        None
    )

    st.session_state.ai_retry_seconds = (
        None
    )

    draw_next_keyword()

    st.session_state.page = (
        "game"
    )


# ============================================================
# 제출 준비
# ============================================================

def prepare_submission(
    image_data,
    timed_out=False,
):

    current_seq = (
        st.session_state.draw_seq
    )

    # 같은 문제 중복 제출 방지
    if (
        st.session_state.submitted_seq
        ==
        current_seq
    ):

        return

    st.session_state.submitted_seq = (
        current_seq
    )

    # --------------------------------------------------------
    # 현재 그림 저장
    # --------------------------------------------------------

    if image_data is not None:

        try:

            snapshot = (
                canvas_array_to_pil(
                    image_data
                )
            )

        except Exception as exc:

            print(
                "[Canvas 변환 오류]",
                str(exc),
            )

            snapshot = (
                blank_canvas_image()
            )

    else:

        snapshot = (
            blank_canvas_image()
        )

    st.session_state.pending_image_bytes = (
        pil_to_png_bytes(
            snapshot
        )
    )

    st.session_state.pending_item = (
        dict(
            st.session_state.current_item
        )
    )

    st.session_state.pending_timed_out = (
        timed_out
    )

    # AI 호출은 별도 페이지에서 실행
    st.session_state.page = (
        "processing"
    )

    st.rerun()


# ============================================================
# 시작 화면
# ============================================================

def start_screen():

    st.title(
        "🎨 AI 캐치마인드"
    )

    st.caption(
        "제시어를 그림으로 표현하고 "
        "AI가 무엇인지 맞히게 해보세요!"
    )

    with st.expander(
        "🕹️ 게임 방법"
    ):

        st.markdown(
            f"""
- 원하는 카테고리를 선택합니다.
- 제시어를 보고 **{ROUND_LIMIT_SEC}초 안에** 그림을 그립니다.
- **제출하기**를 누르면 AI가 그림을 분석합니다.
- 어려운 문제는 **패스**할 수 있습니다.
- 패스는 한 게임에 최대 **{MAX_PASSES}회** 사용할 수 있습니다.
- 패스한 문제는 문제 수에 포함되지 않습니다.
- 총 **{TOTAL_ROUNDS}문제**를 완료하면 결과를 확인합니다.
"""
        )

    names = list(
        CATEGORIES.keys()
    )

    for i in range(
        0,
        len(names),
        3
    ):

        row = names[
            i:i + 3
        ]

        columns = st.columns(
            len(row)
        )

        for column, name in zip(
            columns,
            row,
        ):

            with column:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"""
                        <div style="
                            text-align:center;
                            font-size:44px;
                        ">
                            {CATEGORIES[name]}
                        </div>

                        <div style="
                            text-align:center;
                            font-size:19px;
                            font-weight:700;
                            margin-bottom:8px;
                        ">
                            {name}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if st.button(
                        "시작하기",
                        key=f"start_{name}",
                        use_container_width=True,
                    ):

                        start_new_game(
                            name
                        )

                        st.rerun()


# ============================================================
# 타이머
# ============================================================

@st.fragment(
    run_every=1.0
)
def timer_fragment():

    if (
        st.session_state.page
        !=
        "game"
    ):

        return

    start_time = (
        st.session_state.round_start_time
    )

    if start_time is None:

        return

    elapsed = (
        time.time()
        -
        start_time
    )

    remaining = max(
        0,
        ROUND_LIMIT_SEC
        -
        int(elapsed),
    )

    color = (
        "#D32F2F"
        if remaining <= 10
        else "#666666"
    )

    st.markdown(
        f"""
        <div
            class="timer-box"
            style="color:{color};"
        >
            ⏱ 남은 시간 {remaining}초
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(
        remaining
        /
        ROUND_LIMIT_SEC
    )

    # --------------------------------------------------------
    # 시간 종료
    # --------------------------------------------------------

    if remaining <= 0:

        current_seq = (
            st.session_state.draw_seq
        )

        if (
            st.session_state.timeout_seq
            !=
            current_seq
        ):

            st.session_state.timeout_seq = (
                current_seq
            )

            st.rerun()


# ============================================================
# 게임 화면
# ============================================================

def game_screen():

    item = (
        st.session_state.current_item
    )

    keyword = (
        item["키워드"]
    )

    elapsed = (
        time.time()
        -
        st.session_state.round_start_time
    )

    remaining = max(
        0,
        ROUND_LIMIT_SEC
        -
        int(elapsed),
    )

    time_is_up = (
        remaining <= 0
    )

    # --------------------------------------------------------
    # 상단
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="game-info">
            {st.session_state.category}
            ·
            {st.session_state.round_idx + 1} / {TOTAL_ROUNDS}
            ·
            남은 패스 {MAX_PASSES - st.session_state.passes_used}회
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="keyword-box">
            ✏️ {keyword}
        </div>
        """,
        unsafe_allow_html=True,
    )

    timer_fragment()

    # ========================================================
    # 시간 종료
    # ========================================================

    if time_is_up:

        st.warning(
            "⏰ 시간이 다 됐어요! "
            "마지막 그림으로 자동 제출합니다."
        )

        if (
            st.session_state.submitted_seq
            !=
            st.session_state.draw_seq
        ):

            prepare_submission(
                st.session_state.last_canvas_data,
                timed_out=True,
            )

        return

    # ========================================================
    # 색상 팔레트
    # ========================================================

    st.caption(
        "🎨 색상"
    )

    colors = [
        ("⚫", "#000000"),
        ("🔴", "#E53935"),
        ("🔵", "#1E88E5"),
        ("🟢", "#43A047"),
    ]

    palette_columns = (
        st.columns(4)
    )

    for column, (
        label,
        color,
    ) in zip(
        palette_columns,
        colors,
    ):

        with column:

            if st.button(
                label,
                key=f"color_{color}",
                use_container_width=True,
            ):

                st.session_state.stroke_color = (
                    color
                )

                st.rerun()

    tool1, tool2 = (
        st.columns(2)
    )

    with tool1:

        st.color_picker(
            "다른 색",
            key="stroke_color",
        )

    with tool2:

        st.slider(
            "펜 굵기",
            min_value=3,
            max_value=20,
            key="stroke_width",
        )

    # ========================================================
    # 그림판
    #
    # streamlit-drawable-canvas 0.13.0 공식 API
    # ========================================================

    canvas_result = st_canvas(

        fill_color=(
            "rgba(0, 0, 0, 0)"
        ),

        stroke_width=(
            st.session_state.stroke_width
        ),

        stroke_color=(
            st.session_state.stroke_color
        ),

        background_color="#FFFFFF",

        height=CANVAS_HEIGHT,

        width=CANVAS_WIDTH,

        drawing_mode="freedraw",

        update_streamlit=True,

        # 0.13.0에서 image_data 사용 시 필수
        return_image_data=True,

        key=(
            f"canvas_"
            f"{st.session_state.draw_seq}"
        ),
    )

    # --------------------------------------------------------
    # 현재 캔버스 데이터 즉시 확보
    # --------------------------------------------------------

    current_canvas_data = (
        st.session_state.last_canvas_data
    )

    try:

        if (
            canvas_result is not None
            and
            canvas_result.image_data
            is not None
        ):

            current_canvas_data = (
                canvas_result.image_data.copy()
            )

            st.session_state.last_canvas_data = (
                current_canvas_data
            )

    except Exception as exc:

        print(
            "[Canvas image_data 오류]",
            type(exc).__name__,
            str(exc),
        )

    # ========================================================
    # 버튼
    # ========================================================

    remaining_passes = (
        MAX_PASSES
        -
        st.session_state.passes_used
    )

    pass_col, submit_col = (
        st.columns(
            [1, 1.5]
        )
    )

    with pass_col:

        pass_clicked = st.button(
            f"🙅 패스 ({remaining_passes})",
            disabled=(
                remaining_passes <= 0
            ),
            use_container_width=True,
        )

    with submit_col:

        submit_clicked = st.button(
            "제출하기 ✅",
            type="primary",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # 패스
    # --------------------------------------------------------

    if pass_clicked:

        st.session_state.passes_used += 1

        draw_next_keyword()

        st.rerun()

    # --------------------------------------------------------
    # 제출
    #
    # 현재 canvas_result를 바로 전달
    # --------------------------------------------------------

    if submit_clicked:

        prepare_submission(
            current_canvas_data,
            timed_out=False,
        )


# ============================================================
# AI 처리 화면
# ============================================================

def processing_screen():

    image_bytes = (
        st.session_state.pending_image_bytes
    )

    item = (
        st.session_state.pending_item
    )

    if (
        image_bytes is None
        or
        item is None
    ):

        st.session_state.page = (
            "game"
        )

        st.rerun()

    st.subheader(
        "🤖 AI가 생각 중입니다"
    )

    st.image(
        image_bytes,
        width=DISPLAY_IMG_WIDTH,
        caption="제출한 그림",
    )

    image = Image.open(
        BytesIO(
            image_bytes
        )
    ).convert(
        "RGB"
    )

    with st.spinner(
        "그림을 분석하고 있어요..."
    ):

        result = ask_ai_guess(
            image,
            st.session_state.category,
        )

    # ========================================================
    # API 오류
    # ========================================================

    if not result[
        "success"
    ]:

        st.session_state.ai_error_type = (
            result.get(
                "type"
            )
        )

        st.session_state.ai_error_message = (
            result.get(
                "message"
            )
        )

        st.session_state.ai_retry_seconds = (
            result.get(
                "retry_seconds"
            )
        )

        st.session_state.page = (
            "ai_error"
        )

        st.rerun()

    # ========================================================
    # 성공
    # ========================================================

    ai_answer = (
        result["answer"]
    )

    correct = is_correct(
        ai_answer,
        item,
    )

    st.session_state.rounds.append(
        {
            "keyword": item["키워드"],
            "image_bytes": image_bytes,
            "ai_answer": ai_answer,
            "correct": correct,
            "timed_out": (
                st.session_state.pending_timed_out
            ),
        }
    )

    st.session_state.round_idx += 1

    st.session_state.page = (
        "grading"
    )

    st.rerun()


# ============================================================
# AI 오류 화면
# ============================================================

def ai_error_screen():

    st.title(
        "⚠️ AI 연결 안내"
    )

    if (
        st.session_state.pending_image_bytes
        is not None
    ):

        st.image(
            st.session_state.pending_image_bytes,
            width=DISPLAY_IMG_WIDTH,
            caption="제출한 그림",
        )

    error_type = (
        st.session_state.ai_error_type
    )

    message = (
        st.session_state.ai_error_message
        or
        "오류 내용을 확인할 수 없습니다."
    )

    # --------------------------------------------------------
    # 429
    # --------------------------------------------------------

    if error_type == "quota":

        seconds = (
            st.session_state.ai_retry_seconds
            or
            60
        )

        st.warning(
            "⏳ AI 무료 사용량 제한에 도달했습니다."
        )

        st.info(
            f"약 {seconds}초 후 "
            f"'AI 다시 호출'을 눌러주세요."
        )

    # --------------------------------------------------------
    # 503
    # --------------------------------------------------------

    elif error_type == "server":

        st.warning(
            "🌐 Gemini 서버가 현재 혼잡합니다."
        )

        st.info(
            message
        )

    # --------------------------------------------------------
    # 404
    # --------------------------------------------------------

    elif error_type == "model":

        st.error(
            "🤖 Gemini 모델 오류입니다."
        )

        st.code(
            message
        )

    # --------------------------------------------------------
    # API KEY / 권한
    # --------------------------------------------------------

    elif error_type == "configuration":

        st.error(
            "🔑 Gemini API 설정 오류입니다."
        )

        # 실제 오류를 화면에서 볼 수 있게 함
        st.code(
            message
        )

    # --------------------------------------------------------
    # 400 요청 오류
    # --------------------------------------------------------

    elif error_type == "request":

        st.error(
            "📨 Gemini API 요청 오류입니다."
        )

        st.code(
            message
        )

    else:

        st.error(
            "AI 호출 중 오류가 발생했습니다."
        )

        st.code(
            message
        )

    st.write("")

    # --------------------------------------------------------
    # 같은 그림 그대로 다시 호출
    # --------------------------------------------------------

    if st.button(
        "🔄 AI 다시 호출",
        type="primary",
        use_container_width=True,
    ):

        st.session_state.page = (
            "processing"
        )

        st.rerun()

    # --------------------------------------------------------
    # 그림 다시 그리기
    # --------------------------------------------------------

    if st.button(
        "🎨 그림 다시 그리기",
        use_container_width=True,
    ):

        st.session_state.submitted_seq = (
            None
        )

        st.session_state.timeout_seq = (
            None
        )

        st.session_state.round_start_time = (
            time.time()
        )

        st.session_state.page = (
            "game"
        )

        st.rerun()


# ============================================================
# 채점 화면
# ============================================================

def grading_screen():

    result = (
        st.session_state.rounds[
            -1
        ]
    )

    if result[
        "correct"
    ]:

        st.markdown(
            """
            <div class="correct-box">

                <div style="
                    font-size:30px;
                    font-weight:900;
                    color:#15803d;
                ">
                    ✅ 정답이에요!
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="wrong-box">

                <div style="
                    font-size:30px;
                    font-weight:900;
                    color:#b91c1c;
                ">
                    ❌ 아쉬워요
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    st.image(
        result["image_bytes"],
        width=DISPLAY_IMG_WIDTH,
    )

    st.markdown(
        f"""
        <div style="
            text-align:center;
            margin-top:10px;
        ">

            <div class="answer-label">
                🎯 정답
            </div>

            <div class="answer-text">
                {result['keyword']}
            </div>

            <div class="answer-label">
                🤖 AI의 대답
            </div>

            <div class="answer-text">
                {result['ai_answer']}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.get(
        "timed_out"
    ):

        st.info(
            "⏰ 제한시간 종료 후 "
            "자동 제출된 그림입니다."
        )

    is_last = (
        st.session_state.round_idx
        >=
        TOTAL_ROUNDS
    )

    label = (
        "🏆 결과 보기 →"
        if is_last
        else
        "다음 문제 →"
    )

    if st.button(
        label,
        type="primary",
        use_container_width=True,
    ):

        if is_last:

            st.session_state.page = (
                "result"
            )

        else:

            draw_next_keyword()

            st.session_state.page = (
                "game"
            )

        st.rerun()


# ============================================================
# 결과 화면
# ============================================================

def result_screen():

    st.title(
        "🏆 게임 결과"
    )

    st.subheader(
        f"{CATEGORIES.get(st.session_state.category, '')} "
        f"{st.session_state.category}"
    )

    correct_count = sum(
        1
        for result
        in st.session_state.rounds
        if result["correct"]
    )

    st.metric(
        "맞힌 개수",
        f"{correct_count} / "
        f"{len(st.session_state.rounds)}",
    )

    for index, result in enumerate(
        st.session_state.rounds,
        start=1,
    ):

        with st.container(
            border=True
        ):

            col1, col2 = (
                st.columns(
                    [1, 1.6]
                )
            )

            with col1:

                st.image(
                    result["image_bytes"],
                    width=THUMB_IMG_WIDTH,
                )

            with col2:

                if result[
                    "correct"
                ]:

                    st.markdown(
                        f"### ✅ {index}번 정답"
                    )

                else:

                    st.markdown(
                        f"### ❌ {index}번 오답"
                    )

                st.markdown(
                    f"🎯 정답: "
                    f"**{result['keyword']}**"
                )

                st.markdown(
                    f"🤖 AI: "
                    f"**{result['ai_answer']}**"
                )

    st.write("")

    if st.button(
        "🔁 처음부터 다시 하기",
        type="primary",
        use_container_width=True,
    ):

        for key in list(
            st.session_state.keys()
        ):

            del st.session_state[
                key
            ]

        st.rerun()


# ============================================================
# 페이지 라우팅
# ============================================================

if (
    st.session_state.page
    ==
    "start"
):

    start_screen()


elif (
    st.session_state.page
    ==
    "game"
):

    game_screen()


elif (
    st.session_state.page
    ==
    "processing"
):

    processing_screen()


elif (
    st.session_state.page
    ==
    "ai_error"
):

    ai_error_screen()


elif (
    st.session_state.page
    ==
    "grading"
):

    grading_screen()


elif (
    st.session_state.page
    ==
    "result"
):

    result_screen()
