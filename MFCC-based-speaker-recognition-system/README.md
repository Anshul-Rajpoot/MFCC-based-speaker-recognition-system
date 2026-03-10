# 🎤 MFCC-Based Speaker Recognition System
**Python | NumPy | SciPy | Librosa | Streamlit | DSP | ML-ready**

<img width="1886" height="893" alt="image" src="https://github.com/user-attachments/assets/46e3f62f-27f0-4e68-ace5-4b214f455a1b" />

---

## 📌 Overview
This project implements an **MFCC-based Speaker Recognition System** that analyzes and compares human voices using **Digital Signal Processing (DSP)** techniques and **statistical feature engineering**.

The system provides an **interactive Streamlit dashboard** to visualize audio signals, MFCC features, and speaker similarity scores.

### Applications
- Speaker recognition & verification  
- Voice-based authentication systems  
- Learning DSP & speech processing  
- Internship & placement demonstrations  

---

## ⚙️ System Architecture
Audio Input (.wav)
↓
Pre-emphasis
↓
Framing + Windowing
↓
FFT → Power Spectrum
↓
Mel Filter Bank
↓
Log Compression
↓
DCT → MFCC
↓
Δ & ΔΔ Features
↓
Statistical Aggregation
↓
Speaker Similarity (Cosine Distance)


---

## ✨ Key Features

### 🔹 MFCC Feature Extraction (From Scratch)
- Pre-emphasis filtering  
- Frame blocking & Hamming window  
- FFT & power spectrum computation  
- Mel filter bank energy calculation  
- Log scaling & DCT for MFCCs  

### 🔹 Advanced Speech Features
- MFCC (static coefficients)  
- Delta (Δ) and Delta-Delta (ΔΔ) features  
- Short-time Energy  
- Zero Crossing Rate (ZCR)  
- Fixed-length statistical feature vectors (mean + std)

### 🔹 Interactive Streamlit Dashboard
- Time-domain waveform visualization  
- Spectrogram (time–frequency analysis)  
- MFCC heatmaps  
- Feature variance analysis  
- Real-time parameter tuning  

### 🔹 Speaker Similarity & Verification
- Upload two voice samples  
- Cosine similarity-based comparison  
- Decision: **Same Speaker / Different Speaker**

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Anshul-Rajpoot/MFCC-based-speaker-recognition-system.git
cd MFCC-based-speaker-recognition-system
