# QRS-Detection-and-ECG-Signal-Denoising-using-Digital-Filters
""" ECG Signal Processing Project QRS Detection and ECG Signal Denoising using Digital Filters 
Steps:  
1. Data Acquisition (Local mitdb_data folder)
2. 2. Preprocessing (high-pass + notch filter + normalization)
3. Bandpass Filtering (0.5–40 Hz)
4. R-Peak Detection (adaptive thresholding)
5. Feature Extraction (RR intervals, BPM, HRV: SDNN, RMSSD)
6. Frequency Domain Analysis (FFT)
7. Classification (Random Forest — 4-Class Disease Detection)
8. Visualization

Install dependencies first:   pip install wfdb numpy scipy matplotlib scikit-learn """
