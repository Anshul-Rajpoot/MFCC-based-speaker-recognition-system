import streamlit as st
import numpy as np
import librosa
import librosa.display
import io
import soundfile as sf
from datetime import datetime
import uuid
import warnings

from mfcc_module import MFCCProcessor

warnings.filterwarnings("ignore")

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="MFCC Speaker Feature Analysis",
    page_icon="🎤",
    layout="wide"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🎓 Project Info")
st.sidebar.info("**Anshul Rajpoot**\n\nECE | MANIT Bhopal")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ MFCC Parameters")

frame_size = st.sidebar.slider("Frame Size (ms)", 20, 50, 25)
overlap = st.sidebar.slider("Frame Overlap (%)", 0, 90, 50)
frame_stride_ms = int(frame_size * (1 - overlap / 100))

n_fft = st.sidebar.selectbox("FFT Size", [256, 512, 1024], index=1)
n_mels = st.sidebar.slider("Mel Filters", 10, 40, 20)
n_mfcc = st.sidebar.slider("MFCC Coefficients", 5, 20, 13)
pre_emphasis = st.sidebar.slider("Pre-emphasis", 0.90, 0.99, 0.97)

st.sidebar.markdown("---")
st.sidebar.caption(f"Session ID: `{str(uuid.uuid4())[:8]}`")
st.sidebar.caption(datetime.now().strftime("%d %b %Y | %H:%M"))

# --------------------------------------------------
# AUDIO LOADER
# --------------------------------------------------
@st.cache_data
def load_audio(uploaded_file):
    audio_bytes = uploaded_file.read()
    signal, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    return signal, sr

# --------------------------------------------------
# MAIN TITLE
# --------------------------------------------------
st.title("🎤 MFCC-Based Speaker Feature Analysis System")
st.markdown(
    "An **interactive DSP + ML pipeline** for extracting and analyzing "
    "**MFCC features** used in speaker recognition systems."
)

uploaded_file = st.file_uploader("📂 Upload WAV file", type=["wav"])

if uploaded_file is None:
    st.warning("Please upload an audio file to proceed.")
    st.stop()

signal, sr = load_audio(uploaded_file)

mfcc_processor = MFCCProcessor(
    sr=sr,
    frame_size=frame_size,
    frame_stride=frame_stride_ms,
    n_fft=n_fft,
    n_mels=n_mels,
    n_mfcc=n_mfcc,
    pre_emphasis=pre_emphasis
)

result = mfcc_processor.full_pipeline(signal, sr)

# --------------------------------------------------
# TABS
# --------------------------------------------------
tabs = st.tabs([
    "📌 Overview",
    "📈 Signal Analysis",
    "🧠 MFCC Features",
    "🧪 Parameter Study",
    "👤 Speaker Similarity"
])

# --------------------------------------------------
# TAB 1: OVERVIEW
# --------------------------------------------------
with tabs[0]:
    st.subheader("Project Overview")
    st.markdown("""
    **Pipeline**
    ```
    Audio → Framing → FFT → Mel Filters → Log → DCT → MFCC
          → Δ → ΔΔ → Statistical Features → Similarity
    ```
    **Applications**
    - Speaker Recognition
    - Voice Biometrics
    - Speech Analysis
    """)

    st.metric("Audio Duration (sec)", f"{len(signal)/sr:.2f}")
    st.metric("Sample Rate (Hz)", sr)

# --------------------------------------------------
# TAB 2: SIGNAL ANALYSIS
# --------------------------------------------------
with tabs[1]:
    st.subheader("Time & Frequency Domain")

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(mfcc_processor.plot_time_domain(signal, sr, "Time Domain Signal"))

    with col2:
        st.pyplot(mfcc_processor.plot_spectrogram(signal, sr, "Spectrogram"))

# --------------------------------------------------
# TAB 3: MFCC FEATURES
# --------------------------------------------------
with tabs[2]:
    st.subheader("MFCC & Derived Features")

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(mfcc_processor.plot_mfcc(result["mfcc"], sr))

    with col2:
        st.pyplot(mfcc_processor.plot_feature_variance(result["mfcc"]))

    st.markdown("**Feature Vector Dimension:**")
    st.code(result["feature_vector"].shape)

# --------------------------------------------------
# TAB 4: PARAMETER STUDY
# --------------------------------------------------
with tabs[3]:
    st.subheader("Parameter Sensitivity Analysis")

    st.info(
        "Adjust parameters from the sidebar and observe changes "
        "in MFCC patterns and variance."
    )

    st.pyplot(mfcc_processor.plot_mfcc(result["mfcc"], sr))

# --------------------------------------------------
# TAB 5: SPEAKER SIMILARITY
# --------------------------------------------------
with tabs[4]:
    st.subheader("Speaker Similarity (Cosine Distance)")

    st.markdown("Upload another audio sample to compare speakers.")

    uploaded_file_2 = st.file_uploader(
        "Upload second WAV file", type=["wav"], key="speaker2"
    )

    if uploaded_file_2:
        signal2, sr2 = load_audio(uploaded_file_2)

        mfcc_processor_2 = MFCCProcessor(
            sr=sr2,
            frame_size=frame_size,
            frame_stride=frame_stride_ms,
            n_fft=n_fft,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            pre_emphasis=pre_emphasis
        )

        result2 = mfcc_processor_2.full_pipeline(signal2, sr2)

        similarity = MFCCProcessor.speaker_similarity(
            result["feature_vector"],
            result2["feature_vector"]
        )

        st.metric("Speaker Similarity Score", f"{similarity:.3f}")

        if similarity > 0.8:
            st.success("Likely SAME speaker")
        else:
            st.error("Likely DIFFERENT speakers")
