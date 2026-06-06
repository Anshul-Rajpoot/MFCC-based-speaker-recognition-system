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
st.sidebar.markdown("""
<div style="
padding:15px;
border-radius:12px;
background:linear-gradient(135deg,#1e3c72,#2a5298);
color:white;
text-align:center;
">

<h3>👨‍💻 Developer</h3>

<b style="font-size:18px;">
Anshul Rajpoot
</b>

<br><br>

📘 Scholar No<br>
<code>2311401168</code>

<br><br>

🎓 Electronics & Communication Engineering

🏛️ MANIT Bhopal

</div>
""", unsafe_allow_html=True)
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
st.markdown("""
### 🎯 Project Objective

This project performs speech analysis using
**Mel Frequency Cepstral Coefficients (MFCC)** and compares
speakers using **Cosine Similarity**.

The system provides:

- Speech Visualization
- MFCC Extraction
- Feature Analysis
- Speaker Similarity Detection
""")

uploaded_file = st.file_uploader("📂 Upload WAV file", type=["wav"])
if uploaded_file:
    st.audio(uploaded_file)

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
    "👤 Speaker Similarity"
])

# --------------------------------------------------
# TAB 1: OVERVIEW
# --------------------------------------------------

st.markdown("""
### ⚙️ Current Parameters

- Frame Size: {} ms
- Frame Overlap: {} %
- FFT Size: {}
- Mel Filters: {}
- MFCC Coefficients: {}
- Pre-emphasis: {}
""".format(
    frame_size,
    overlap,
    n_fft,
    n_mels,
    n_mfcc,
    pre_emphasis
))

with tabs[0]:
    st.subheader("Project Overview")

    st.markdown("""
### 🔄 Processing Pipeline

Audio → Pre-emphasis → Framing → Windowing → FFT
→ Mel Filter Bank → Log → DCT → MFCC
→ Delta → Delta-Delta → Feature Vector
→ Speaker Similarity

### 🚀 Applications

- Speaker Recognition
- Voice Biometrics
- Speech Analysis
- Audio Analytics
""")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Duration",
            f"{len(signal)/sr:.2f} sec"
        )

    with col2:
        st.metric(
            "Sample Rate",
            f"{sr} Hz"
        )

    with col3:
        st.metric(
            "MFCC Features",
            n_mfcc
        )
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
        st.pyplot(
            mfcc_processor.plot_mfcc(
                result["mfcc"], sr
            )
        )

    with col2:
        st.pyplot(
            mfcc_processor.plot_feature_variance(
                result["mfcc"]
            )
        )

    st.info(
        f"Feature Vector Size: "
        f"{result['feature_vector'].shape[0]}"
    )

    st.markdown("### Extracted Features")

    feature_col1, feature_col2, feature_col3 = st.columns(3)

    feature_col1.metric(
        "MFCC",
        result["mfcc"].shape[1]
    )

    feature_col2.metric(
        "Delta",
        result["delta"].shape[1]
    )

    feature_col3.metric(
        "Delta-Delta",
        result["delta2"].shape[1]
    )

# --------------------------------------------------
# TAB 4: SPEAKER SIMILARITY
# --------------------------------------------------
with tabs[3]:
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

        st.metric(
    "Speaker Similarity Score",
    f"{similarity:.3f}"
)

st.progress(float(similarity))

       if similarity >= 0.9:
    st.success(
        "Very High Similarity"
    )

elif similarity >= 0.8:
    st.success(
        "Likely SAME Speaker"
    )

elif similarity >= 0.6:
    st.warning(
        "Moderate Similarity"
    )

else:
    st.error(
        "Likely DIFFERENT Speaker"
    )



st.markdown("---")

st.caption(
    "🎤 MFCC-Based Speaker Feature Analysis System | "
    "Developed by Anshul Rajpoot | MANIT Bhopal"
)
