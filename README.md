# 🎤 MFCC-Based Speaker Recognition System

**Python | NumPy | SciPy | Librosa | Streamlit | DSP | ML-ready**

<img width="1886" height="893" alt="MFCC Speaker Recognition System Screenshot" src="https://github.com/user-attachments/assets/46e3f62f-27f0-4e68-ace5-4b214f455a1b" />

---

## 🚀 Live Demo

🔗 **Try the App:**
https://mfcc-based-speaker-recognition-system-ykjnr2bja8rdzdds3zmpc4.streamlit.app/

---

## 📌 Overview

This project implements an **MFCC-based Speaker Recognition System** that analyzes and compares human voices using **Digital Signal Processing (DSP)** techniques and **statistical feature engineering**.

The system extracts **Mel-Frequency Cepstral Coefficients (MFCCs)** from speech signals and compares speakers using **cosine similarity**.

An **interactive Streamlit dashboard** allows users to upload audio files and visualize signal characteristics, MFCC features, and similarity scores between speakers.

---

## 🧠 Applications

* Speaker recognition & verification
* Voice-based authentication systems
* Learning **Digital Signal Processing (DSP)** and speech analysis
* Demonstration project for **machine learning / signal processing internships**

---

## ⚙️ System Architecture

```
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
```

---

## ✨ Key Features

### 🔹 MFCC Feature Extraction (From Scratch)

* Pre-emphasis filtering
* Frame blocking & Hamming window
* FFT & power spectrum computation
* Mel filter bank energy calculation
* Log scaling and **Discrete Cosine Transform (DCT)** for MFCC generation

---

### 🔹 Advanced Speech Features

* **MFCC coefficients**
* **Delta (Δ) and Delta-Delta (ΔΔ) features**
* **Short-Time Energy**
* **Zero Crossing Rate (ZCR)**
* Fixed-length statistical feature vectors *(mean + standard deviation)*

---

### 🔹 Interactive Streamlit Dashboard

* Time-domain waveform visualization
* Spectrogram for time–frequency analysis
* MFCC heatmap visualization
* Feature variance analysis
* Interactive audio comparison interface

---

### 🔹 Speaker Similarity & Verification

* Upload two voice samples
* Extract MFCC features from both signals
* Compute **cosine similarity between speaker feature vectors**
* System outputs:

```
Same Speaker
or
Different Speaker
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Anshul-Rajpoot/MFCC-based-speaker-recognition-system.git
cd MFCC-based-speaker-recognition-system
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
🎤 MFCC-Based Speaker Recognition
│
├── Live Demo Badge
├── Screenshot
├── System Architecture
├── Features
├── How to Run
└── Author
```

---

## 🛠 Technologies Used

* **Python**
* **NumPy**
* **SciPy**
* **Librosa**
* **Streamlit**
* **Digital Signal Processing (DSP)**
* **Feature Engineering for Speech Processing**

---

## 📌 Core Concept

The system extracts **Mel-Frequency Cepstral Coefficients (MFCCs)** from audio signals, which represent the **spectral characteristics of human speech**.

Speaker comparison is performed using **cosine similarity** between feature vectors derived from the MFCC coefficients and their statistical representations.

---

## 👨‍💻 Author

**Anshul Rajpoot**
ECE Undergraduate | Data Science & Machine Learning Enthusiast

🔗 GitHub:
https://github.com/Anshul-Rajpoot

---

⭐ If you like this project, consider **starring the repository**!
