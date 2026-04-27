
# ── Imports ───────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import os
import glob
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore")

try:
    import wfdb
except ImportError:
    print("[!] wfdb not installed. Run: pip install wfdb")
    exit(1)


#  STEP 1 — DATA ACQUISITION (LOCAL)

def load_local_ecg(record_name: str, data_dir: str = "mitdb_data", duration_s: int = 30) -> tuple:
    """
    Loads an ECG record entirely from the local directory.
    Requires the .dat and .hea files to be present in data_dir.
    """
    print(f"[1] Loading local record '{record_name}' from '{data_dir}/'...")
    record_path = os.path.join(data_dir, record_name)
    
    if not os.path.exists(record_path + ".dat"):
        raise FileNotFoundError(f"Missing local file: {record_path}.dat")
        
    record = wfdb.rdrecord(record_path, sampfrom=0, sampto=duration_s * 360)
    ecg = record.p_signal[:, 0]
    fs  = record.fs
    print(f"    Loaded {len(ecg)} samples @ {fs} Hz ({duration_s}s)")
    return ecg, fs


#  STEP 2 — PREPROCESSING

def remove_baseline_wander(ecg: np.ndarray, fs: int,
                            cutoff_hz: float = 0.5) -> np.ndarray:
    """High-pass Butterworth filter to remove baseline drift."""
    b, a = signal.butter(4, cutoff_hz / (fs / 2), btype="high")
    return signal.filtfilt(b, a, ecg)


def remove_powerline_noise(ecg: np.ndarray, fs: int,
                           freq_hz: float = 50.0, Q: float = 30.0) -> np.ndarray:
    """IIR notch filter at 50 Hz (or 60 Hz for USA)."""
    b, a = signal.iirnotch(freq_hz, Q, fs)
    return signal.filtfilt(b, a, ecg)


def normalize_signal(ecg: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization."""
    return (ecg - np.mean(ecg)) / np.std(ecg)


def preprocess(ecg_raw: np.ndarray, fs: int) -> np.ndarray:
    print("[2] Preprocessing: baseline removal → notch filter → normalization")
    ecg = remove_baseline_wander(ecg_raw, fs)
    ecg = remove_powerline_noise(ecg, fs)
    ecg = normalize_signal(ecg)
    return ecg


#  STEP 3 — BANDPASS FILTERING (0.5–40 Hz)

def bandpass_filter(ecg: np.ndarray, fs: int,
                    low_hz: float = 0.5, high_hz: float = 40.0,
                    order: int = 4) -> np.ndarray:
    """4th-order Butterworth bandpass — retains clinically relevant content."""
    print(f"[3] Bandpass filtering: {low_hz}–{high_hz} Hz")
    nyq = fs / 2
    b, a = signal.butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return signal.filtfilt(b, a, ecg)

#  STEP 4 — R-PEAK DETECTION (adaptive threshold)

def detect_r_peaks(ecg_filtered: np.ndarray, fs: int) -> np.ndarray:
    """
    Upgraded R-peak detector combining Pan-Tompkins integration
    with robust scipy prominence filtering.
    """
    print("[4] Detecting R-peaks (hybrid prominence threshold)...")

    diff    = np.diff(ecg_filtered)
    squared = diff ** 2
    win     = int(0.150 * fs)
    mwi     = np.convolve(squared, np.ones(win) / win, mode="same")

    peaks, _ = signal.find_peaks(mwi,
                                 distance=int(0.2 * fs),
                                 prominence=np.mean(mwi) * 1.5)

    true_peaks = []
    for p in peaks:
        search = slice(max(0, p - int(0.05 * fs)),
                       min(len(ecg_filtered), p + int(0.05 * fs)))
        if len(ecg_filtered[search]) > 0:
            true_r = np.argmax(ecg_filtered[search]) + search.start
            true_peaks.append(true_r)

    true_peaks = np.array(true_peaks)
    print(f"    Detected {len(true_peaks)} R-peaks")
    return true_peaks

#  STEP 5 — FEATURE EXTRACTION

def extract_features(r_peaks: np.ndarray, fs: int) -> dict:
    """
    Computes:
      RR intervals (ms), mean BPM, SDNN, RMSSD, pNN50,
      and coefficient of variation.
    """
    print("[5] Extracting time-domain HRV features...")

    rr_samples = np.diff(r_peaks)
    rr_ms      = (rr_samples / fs) * 1000

    if len(rr_ms) < 2:
        raise ValueError("Too few beats detected to compute features.")

    mean_rr  = np.mean(rr_ms)
    bpm      = 60_000 / mean_rr
    sdnn     = np.std(rr_ms, ddof=1)
    rmssd    = np.sqrt(np.mean(np.diff(rr_ms) ** 2))
    pnn50    = 100.0 * np.sum(np.abs(np.diff(rr_ms)) > 50) / len(rr_ms)
    cv_rr    = sdnn / mean_rr

    features = {
        "mean_rr_ms" : round(mean_rr,  2),
        "bpm"        : round(bpm,      2),
        "sdnn_ms"    : round(sdnn,     2),
        "rmssd_ms"   : round(rmssd,    2),
        "pnn50_pct"  : round(pnn50,    2),
        "cv_rr"      : round(cv_rr,    4),
        "num_beats"  : len(r_peaks),
    }

    return features

#  STEP 6 — FREQUENCY DOMAIN ANALYSIS (FFT)

def frequency_analysis(ecg_filtered: np.ndarray, fs: int,
                        rr_ms: np.ndarray = None) -> dict:
    """
    Two analyses:
      A) FFT of the ECG waveform — shows dominant frequency content.
      B) HRV frequency-domain (if RR series provided):
         VLF (<0.04 Hz), LF (0.04–0.15 Hz), HF (0.15–0.40 Hz), LF/HF ratio.
    """
    print("[6] Frequency-domain analysis (FFT)...")

    N    = len(ecg_filtered)
    yf   = np.abs(fft(ecg_filtered))[:N // 2]
    xf   = fftfreq(N, 1 / fs)[:N // 2]

    freq_results = {"ecg_fft_freqs": xf, "ecg_fft_mag": yf}

    if rr_ms is not None and len(rr_ms) > 10:
        rr_s         = rr_ms / 1000
        cumtime      = np.cumsum(np.insert(rr_s, 0, 0))
        interp_rate  = 4.0
        t_uniform    = np.arange(cumtime[0], cumtime[-1], 1 / interp_rate)
        rr_uniform   = np.interp(t_uniform, cumtime[:-1], rr_s)

        freqs, psd = signal.welch(rr_uniform, fs=interp_rate,
                                   nperseg=min(256, len(rr_uniform)),
                                   noverlap=None)

        vlf_mask = freqs < 0.04
        lf_mask  = (freqs >= 0.04)  & (freqs < 0.15)
        hf_mask  = (freqs >= 0.15)  & (freqs < 0.40)

        vlf_power = np.trapz(psd[vlf_mask], freqs[vlf_mask]) if vlf_mask.any() else 0
        lf_power  = np.trapz(psd[lf_mask],  freqs[lf_mask])  if lf_mask.any()  else 0
        hf_power  = np.trapz(psd[hf_mask],  freqs[hf_mask])  if hf_mask.any()  else 0
        lf_hf     = lf_power / hf_power if hf_power > 0 else np.nan

        freq_results.update({
            "hrv_freqs"  : freqs,
            "hrv_psd"    : psd,
            "vlf_power"  : round(vlf_power, 4),
            "lf_power"   : round(lf_power,  4),
            "hf_power"   : round(hf_power,  4),
            "lf_hf_ratio": round(lf_hf,     4) if not np.isnan(lf_hf) else "N/A",
        })

    return freq_results


#  STEP 7 — BEAT-LEVEL CLASSIFICATION (4-Class)
#  UPDATED: 6 Features (Timing + Amp + Energy + QRS Width)
#  UPDATED: Custom class weights to fix PAC Precision collapse

def get_qrs_width(ecg_filt, peak_idx, win):
    """Calculates Full-Width at Half-Maximum (FWHM) of the QRS complex."""
    peak_val = ecg_filt[peak_idx]
    half_max = peak_val * 0.5
    
    left_idx = peak_idx
    while left_idx > peak_idx - win and ecg_filt[left_idx] > half_max: 
        left_idx -= 1
        
    right_idx = peak_idx
    while right_idx < peak_idx + win and ecg_filt[right_idx] > half_max: 
        right_idx += 1
        
    return right_idx - left_idx


def build_real_dataset_and_train(data_dir="mitdb_data", fs=360):
    print("[7] Building BALANCED BEAT-LEVEL dataset with 6 Features...")

    search_pattern = os.path.join(data_dir, "*.dat")
    record_paths   = [f.replace(".dat", "") for f in glob.glob(search_pattern)]
    print(f"    -> Found {len(record_paths)} local records to scan")

    NORMAL_SYMS = {"N", "L", "R", "e", "j"}
    PVC_SYMS    = {"V", "E"}
    PAC_SYMS    = {"A", "a", "J", "S"}

    target_names = ["Normal", "Ventricular (PVC)", "Atrial (PAC)", "Bradycardia"]
    feature_keys = ["Pre_RR_ms", "Post_RR_ms", "Local_Mean_RR", "R_Amplitude", "QRS_Energy", "QRS_Width_ms"]

    X_list, y_list = [], []
    win = int(0.05 * fs) # 50ms window

    for record_path in sorted(record_paths):
        try:
            record   = wfdb.rdrecord(record_path, sampfrom=0, sampto=300 * fs)
            ecg_raw  = record.p_signal[:, 0]
            
            ecg_pre  = preprocess(ecg_raw, fs)
            ecg_filt = bandpass_filter(ecg_pre, fs)

            ann      = wfdb.rdann(record_path, "atr", sampfrom=0, sampto=300 * fs)
            peaks    = ann.sample
            symbols  = ann.symbol

            for i in range(1, len(peaks) - 1):
                sym = symbols[i]
                peak_idx = peaks[i]

                if sym in NORMAL_SYMS: candidate = "normal"
                elif sym in PVC_SYMS:  candidate = "pvc"
                elif sym in PAC_SYMS:  candidate = "pac"
                else: continue

                # 1. Timing Features
                pre_rr  = (peaks[i] - peaks[i - 1]) / fs * 1000
                post_rr = (peaks[i + 1] - peaks[i]) / fs * 1000
                mean_rr = (pre_rr + post_rr) / 2.0

                if not (200 < pre_rr < 2000 and 200 < post_rr < 2000): continue
                if peak_idx - win < 0 or peak_idx + win >= len(ecg_filt): continue

                # 2. Morphology Features
                r_amp      = ecg_filt[peak_idx]
                qrs_energy = np.sum(ecg_filt[peak_idx - win : peak_idx + win] ** 2)
                qrs_width  = get_qrs_width(ecg_filt, peak_idx, win) / fs * 1000 # in ms

                # Assign final label
                if candidate == "pvc": label = 1
                elif candidate == "pac": label = 2
                else: label = 3 if mean_rr > 1000 else 0

                X_list.append([pre_rr, post_rr, mean_rr, r_amp, qrs_energy, qrs_width])
                y_list.append(label)

        except Exception:
            continue

    X = np.array(X_list)
    y = np.array(y_list)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # THE FIX: Custom tuned weights to stop PAC False Positives.
    # Tells model: PACs are 4x more important than Normal, but not 30x (which breaks precision).
    tuned_weights = {0: 1, 1: 3, 2: 4, 3: 1}
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42,
                                  max_depth=12, class_weight=tuned_weights, n_jobs=-1)
    clf.fit(X_train_s, y_train)

    y_pred   = clf.predict(X_test_s)
    accuracy = clf.score(X_test_s, y_test)

    print("\n[DISEASE CLASSIFICATION RESULTS]")
    print(f"    Total Beats Used  : {len(X):,}")
    print(f"    Test Accuracy     : {accuracy:.2%}")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return {
        "clf"         : clf,
        "scaler"      : scaler,
        "y_test"      : y_test,
        "y_pred"      : y_pred,
        "accuracy"    : accuracy,
        "feat_keys"   : feature_keys,
        "target_names": target_names,
        "cm"          : confusion_matrix(y_test, y_pred),
    }


#  STEP 8 — VISUALIZATION

DISEASE_UI = {
    0: ("NORMAL RHYTHM",                "#39ff14"),   
    1: ("VENTRICULAR ARRHYTHMIA (PVC)", "#ff4f4f"),   
    2: ("ATRIAL ARRHYTHMIA (PAC/AFIB)", "#ffc400"),   
    3: ("BRADYCARDIA (SLOW RHYTHM)",    "#00d4ff"),   
}


def visualize_all(ecg_raw, ecg_filtered, r_peaks, features,
                  freq_results, clf_results, fs,
                  t_window_s: float = 10.0):
    print("[8] Plotting results...")

    t   = np.arange(len(ecg_raw)) / fs
    win = int(t_window_s * fs)

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

    SIGNAL_COLOR   = "#00d4ff"
    FILTERED_COLOR = "#39ff14"
    PEAK_COLOR     = "#ff4f4f"
    TGRAM_COLOR    = "#c778dd"
    FFT_COLOR      = "#ffc400"
    PSD_COLOR      = "#00ffc3"
    AXES_BG        = "#1a1d2e"

    def style_ax(ax, title):
        ax.set_facecolor(AXES_BG)
        ax.spines[:].set_color("#3a3d55")
        ax.tick_params(colors="#9ca3af", labelsize=8)
        ax.set_title(title, color="#e5e7eb", fontsize=11, pad=8)
        ax.xaxis.label.set_color("#9ca3af")
        ax.yaxis.label.set_color("#9ca3af")
        ax.grid(True, color="#2a2d42", linewidth=0.5, linestyle="--")

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t[:win], ecg_raw[:win], color=SIGNAL_COLOR, lw=0.6, alpha=0.85)
    style_ax(ax1, "Raw ECG Signal")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")

    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(t[:win], ecg_filtered[:win], color=FILTERED_COLOR, lw=0.7, label="Filtered")
    peaks_in_window = r_peaks[r_peaks < win]
    ax2.scatter(peaks_in_window / fs, ecg_filtered[peaks_in_window],
                color=PEAK_COLOR, s=40, zorder=5, label="R-peaks")
    style_ax(ax2, "Filtered ECG + R-Peaks")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(fontsize=8, facecolor="#1a1d2e", labelcolor="#e5e7eb", edgecolor="#3a3d55")

    ax3 = fig.add_subplot(gs[2, :2])
    rr_ms_series = np.diff(r_peaks) / fs * 1000
    ax3.plot(range(len(rr_ms_series)), rr_ms_series,
             color=TGRAM_COLOR, lw=1.2, marker="o", ms=3)
    ax3.axhline(features["mean_rr_ms"], color="#ff7e21",
                lw=1.2, ls="--", label=f"Mean RR = {features['mean_rr_ms']} ms")
    style_ax(ax3, "RR Interval Tachogram")
    ax3.set_xlabel("Beat index")
    ax3.set_ylabel("RR (ms)")
    ax3.legend(fontsize=8, facecolor="#1a1d2e", labelcolor="#e5e7eb", edgecolor="#3a3d55")

    ax4 = fig.add_subplot(gs[0, 2])
    xf   = freq_results["ecg_fft_freqs"]
    yf   = freq_results["ecg_fft_mag"]
    mask = xf <= 60
    ax4.plot(xf[mask], yf[mask], color=FFT_COLOR, lw=0.8)
    ax4.fill_between(xf[mask], yf[mask], alpha=0.2, color=FFT_COLOR)
    style_ax(ax4, "ECG FFT Spectrum")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("|FFT|")

    ax5 = fig.add_subplot(gs[1, 2])
    if "hrv_psd" in freq_results:
        freqs = freq_results["hrv_freqs"]
        psd   = freq_results["hrv_psd"]
        ax5.semilogy(freqs, psd, color=PSD_COLOR, lw=0.9)
        ax5.fill_between(freqs, psd, alpha=0.15, color=PSD_COLOR)
        ax5.axvspan(0,    0.04, alpha=0.12, color="#c778dd", label="VLF")
        ax5.axvspan(0.04, 0.15, alpha=0.12, color="#ffc400", label="LF")
        ax5.axvspan(0.15, 0.40, alpha=0.12, color="#00d4ff", label="HF")
        ax5.legend(fontsize=7, facecolor="#1a1d2e", labelcolor="#e5e7eb",
                   edgecolor="#3a3d55", loc="upper right")
    style_ax(ax5, "HRV Power Spectral Density")
    ax5.set_xlabel("Frequency (Hz)")
    ax5.set_ylabel("Power")

    ax6 = fig.add_subplot(gs[2, 2])
    cm  = clf_results["cm"]
    disp = ConfusionMatrixDisplay(cm, display_labels=clf_results["target_names"])
    disp.plot(ax=ax6, colorbar=False, cmap="plasma")
    for text in ax6.texts:
        text.set_color("white")
        text.set_fontsize(10)
    style_ax(ax6, f"Confusion Matrix  (acc={clf_results['accuracy']:.0%})")
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=30, ha="right", fontsize=7)
    ax6.set_yticklabels(ax6.get_yticklabels(), fontsize=7)

    pred_idx    = clf_results.get("pred_class", 0)
    diag_label  = clf_results["target_names"][pred_idx]
    _, diag_clr = DISEASE_UI[pred_idx]

    summary = (
        f"  Heart Rate : {features['bpm']} BPM\n"
        f"  SDNN       : {features['sdnn_ms']} ms\n"
        f"  RMSSD      : {features['rmssd_ms']} ms\n"
        f"  pNN50      : {features['pnn50_pct']} %\n"
        f"  Beats      : {features['num_beats']}\n"
        f"  Prediction : {diag_label}  "
        f"({clf_results['confidence']:.1%})"
    )
    fig.text(0.01, 0.01, summary,
             color="#e5e7eb", fontsize=9,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1d2e", edgecolor=diag_clr, alpha=0.9))

    fig.suptitle("ECG Signal Analysis — Group 73", color="#e5e7eb", fontsize=15, fontweight="bold", y=0.98)

    plt.savefig("ecg_analysis_output.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("    Saved → ecg_analysis_output.png")
    plt.show()


#  ADD-ON 1: FEATURE IMPORTANCE

def plot_feature_importance(clf_results):
    print("\n[8B] Generating Feature Importance Chart...")
    clf  = clf_results["clf"]
    keys = clf_results["feat_keys"]
    importance = clf.feature_importances_

    sorted_idx       = np.argsort(importance)
    sorted_keys      = [keys[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")

    bars = ax.barh(sorted_keys, sorted_importance, color="#ffc400", edgecolor="#ffc400", alpha=0.8)

    for bar, val in zip(bars, sorted_importance):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="#e5e7eb", fontsize=9)

    ax.set_xlabel("Gini Importance Score", color="#9ca3af")
    ax.set_title("Random Forest Feature Importance\n(What determines disease class?)", color="#e5e7eb", fontweight="bold")
    ax.tick_params(colors="#9ca3af")
    ax.spines[:].set_color("#3a3d55")
    ax.grid(axis="x", linestyle="--", alpha=0.3, color="#9ca3af")

    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, facecolor=fig.get_facecolor())
    print("    Saved → feature_importance.png")
    plt.show()


#  ADD-ON 2: TRIPLE COMPARISON MONITOR

def triple_live_comparison_monitor(data_dir="mitdb_data", fs=360):
    print("\n[8C] Launching Triple Comparison Monitor...")
    rec_norm, rec_mild, rec_abn  = '100', '103', '200'
    duration = 30

    def load_and_prep(rec_name):
        rec_path = os.path.join(data_dir, rec_name)
        record   = wfdb.rdrecord(rec_path, sampfrom=0, sampto=duration * fs)
        ecg_raw  = record.p_signal[:, 0]
        ecg_pre  = preprocess(ecg_raw, fs)
        ecg_filt = bandpass_filter(ecg_pre, fs)
        import sys, os as _os
        old_stdout = sys.stdout
        sys.stdout = open(_os.devnull, "w")
        r_peaks = detect_r_peaks(ecg_filt, fs)
        sys.stdout = old_stdout
        return ecg_filt, r_peaks

    print("    -> Loading Patient 1 (Normal)...")
    sig1, pks1 = load_and_prep(rec_norm)
    print("    -> Loading Patient 2 (In-between)...")
    sig2, pks2 = load_and_prep(rec_mild)
    print("    -> Loading Patient 3 (Abnormal)...")
    sig3, pks3 = load_and_prep(rec_abn)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor("black")
    fig.canvas.manager.set_window_title("Group 73 - Clinical Comparison")

    titles = [
        "PATIENT 1: NORMAL SINUS RHYTHM (Record 100)",
        "PATIENT 2: BORDERLINE / MILD ARRHYTHMIA (Record 103)",
        "PATIENT 3: SEVERE ARRHYTHMIA [PVCs] (Record 200)",
    ]
    colors = ["#39ff14", "#ffc400", "#ff4f4f"]
    lines, scatters, texts_bpm, texts_status = [], [], [], []
    window_size = int(4 * fs)

    for i, ax in enumerate(axes):
        ax.set_facecolor("black")
        ax.set_xlim(-4, 0)
        sig_i = [sig1, sig2, sig3][i]
        ax.set_ylim(np.min(sig_i) * 1.2, np.max(sig_i) * 1.2)
        ax.axis("off")
        ax.grid(True, color="#003300", linewidth=0.5)
        ax.text(0.01, 0.85, titles[i], transform=ax.transAxes, color=colors[i], fontsize=11, fontweight="bold")

        line,  = ax.plot([], [], lw=1.5, color=colors[i])
        scat   = ax.scatter([], [], color="white", s=40, zorder=5)
        bpm_t  = ax.text(0.88, 0.85, "BPM: --", transform=ax.transAxes, color=colors[i], fontsize=12, fontweight="bold")
        stat_t = ax.text(0.90, 0.15, "", transform=ax.transAxes, color="white", fontsize=10, fontweight="bold")

        lines.append(line)
        scatters.append(scat)
        texts_bpm.append(bpm_t)
        texts_status.append(stat_t)

    def update(frame):
        start_idx = max(0, frame - window_size)
        end_idx   = frame
        sigs, pks = [sig1, sig2, sig3], [pks1, pks2, pks3]
        artists   = []

        for i in range(3):
            y_data = sigs[i][start_idx:end_idx]
            x_data = np.linspace(-len(y_data) / fs, 0, len(y_data))
            lines[i].set_data(x_data, y_data)
            artists.append(lines[i])

            win_peaks = pks[i][(pks[i] >= start_idx) & (pks[i] < end_idx)]
            if len(win_peaks) > 0:
                px = (win_peaks - end_idx) / fs
                py = sigs[i][win_peaks]
                scatters[i].set_offsets(np.c_[px, py])

                past_peaks = pks[i][pks[i] <= end_idx]
                if len(past_peaks) >= 2:
                    rr  = (past_peaks[-1] - past_peaks[-2]) / fs * 1000
                    bpm = 60000 / rr if rr > 0 else 0
                    texts_bpm[i].set_text(f"BPM: {int(bpm)}")
                    if (end_idx - past_peaks[-1]) < 0.1 * fs:
                        texts_status[i].set_text("♥ BEAT")
                        texts_status[i].set_color(colors[i])
                    else:
                        texts_status[i].set_text("")
            else:
                scatters[i].set_offsets(np.empty((0, 2)))
                texts_status[i].set_text("")

            artists.extend([scatters[i], texts_bpm[i], texts_status[i]])

        return artists

    ani = animation.FuncAnimation(fig, update, frames=range(0, duration * fs, 15), interval=30, blit=True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.show()


#  ADD-ON 3: LIVE SCROLLING DISEASE MONITOR

def live_scrolling_monitor_advanced(target_record, clf_results, data_dir="mitdb_data", fs=360):
    print(f"\n[8D] Launching Live Disease Monitor for record {target_record}...")

    clf    = clf_results["clf"]
    scaler = clf_results["scaler"]

    record_path = os.path.join(data_dir, target_record)
    record      = wfdb.rdrecord(record_path, sampfrom=0, sampto=60 * fs)
    ecg_raw     = record.p_signal[:, 0]
    ecg_pre     = preprocess(ecg_raw, fs)
    ecg_filt    = bandpass_filter(ecg_pre, fs)

    import sys, os as _os
    old_stdout = sys.stdout
    sys.stdout = open(_os.devnull, "w")
    r_peaks = detect_r_peaks(ecg_filt, fs)
    sys.stdout = old_stdout

    window_size = int(4 * fs)
    win_energy  = int(0.05 * fs) 

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("black")
    fig.canvas.manager.set_window_title(f"Group 73 — Live Disease Monitor | Record {target_record}")

    ax_ecg = axes[0]
    ax_ecg.set_facecolor("black")
    ax_ecg.set_xlim(-4, 0)
    ax_ecg.set_ylim(np.min(ecg_filt) * 1.3, np.max(ecg_filt) * 1.3)
    ax_ecg.axis("off")
    ax_ecg.grid(True, color="#003300", linewidth=0.5)
    ax_ecg.set_title(f"LIVE ECG MONITOR — Record {target_record}", color="white", fontsize=13, fontweight="bold", pad=10)

    ecg_line,  = ax_ecg.plot([], [], lw=1.5, color="#39ff14")
    peak_scat  = ax_ecg.scatter([], [], color="white", s=50, zorder=5)
    bpm_text   = ax_ecg.text(0.01, 0.90, "BPM: --", transform=ax_ecg.transAxes, color="#39ff14", fontsize=14, fontweight="bold")

    ax_ai = axes[1]
    ax_ai.set_facecolor("#0a0a0a")
    ax_ai.axis("off")

    ai_box = ax_ai.text(
        0.5, 0.5, "AI Diagnosis: INITIALIZING...",
        transform=ax_ai.transAxes, ha="center", va="center",
        fontsize=16, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#1a1d2e", edgecolor="white", linewidth=2)
    )

    def update(frame):
        start_idx = max(0, frame - window_size)
        end_idx   = frame

        y_data = ecg_filt[start_idx:end_idx]
        x_data = np.linspace(-len(y_data) / fs, 0, len(y_data))
        ecg_line.set_data(x_data, y_data)

        peaks_in_win = r_peaks[(r_peaks >= start_idx) & (r_peaks < end_idx)]
        if len(peaks_in_win) > 0:
            px = (peaks_in_win - end_idx) / fs
            py = ecg_filt[peaks_in_win]
            peak_scat.set_offsets(np.c_[px, py])
        else:
            peak_scat.set_offsets(np.empty((0, 2)))

        past_peaks = r_peaks[r_peaks <= end_idx]
        if len(past_peaks) >= 2:
            rr  = (past_peaks[-1] - past_peaks[-2]) / fs * 1000
            bpm = 60000 / rr if rr > 0 else 0
            bpm_text.set_text(f"BPM: {int(bpm)}")

        if len(past_peaks) >= 3:
            peak_idx = past_peaks[-2] 
            pre_rr   = (past_peaks[-2] - past_peaks[-3]) / fs * 1000
            post_rr  = (past_peaks[-1] - past_peaks[-2]) / fs * 1000
            mean_rr  = (pre_rr + post_rr) / 2.0

            if 200 < pre_rr < 2000 and peak_idx - win_energy > 0 and peak_idx + win_energy < len(ecg_filt):
                r_amp      = ecg_filt[peak_idx]
                qrs_energy = np.sum(ecg_filt[peak_idx - win_energy : peak_idx + win_energy] ** 2)
                qrs_width  = get_qrs_width(ecg_filt, peak_idx, win_energy) / fs * 1000

                live_vec = np.array([[pre_rr, post_rr, mean_rr, r_amp, qrs_energy, qrs_width]])
                pred     = clf.predict(scaler.transform(live_vec))[0]
                text_label, color_code = DISEASE_UI[pred]

                ai_box.set_text(f"AI Diagnosis: {text_label}")
                ai_box.set_color(color_code)
                ai_box.get_bbox_patch().set_edgecolor(color_code)

        return ecg_line, peak_scat, bpm_text, ai_box

    ani = animation.FuncAnimation(fig, update, frames=range(0, 60 * fs, 15), interval=30, blit=True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.show()

#  MAIN PIPELINE

def run_pipeline(target_record="100", data_dir="mitdb_data"):
    print("=" * 55)
    print("  ECG Analysis Pipeline — Group 73")
    print("=" * 55)

    fs = 360  

    clf_results = build_real_dataset_and_train(data_dir=data_dir, fs=fs)

    print(f"\n[ANALYZING TARGET RECORD: {target_record}]")
    ecg_raw, fs = load_local_ecg(target_record, data_dir=data_dir)

    ecg_pre      = preprocess(ecg_raw, fs)
    ecg_filtered = bandpass_filter(ecg_pre, fs, low_hz=0.5, high_hz=40.0)
    r_peaks      = detect_r_peaks(ecg_filtered, fs)
    features     = extract_features(r_peaks, fs)
    rr_ms        = np.diff(r_peaks) / fs * 1000
    freq_results = frequency_analysis(ecg_filtered, fs, rr_ms=rr_ms)

    # ── Predict disease for the target record ────────────────────────────────
    beat_preds = []
    win_energy = int(0.05 * fs) 

    for i in range(1, len(r_peaks) - 1):
        peak_idx = r_peaks[i]
        pre_rr  = (r_peaks[i] - r_peaks[i-1]) / fs * 1000
        post_rr = (r_peaks[i+1] - r_peaks[i]) / fs * 1000
        mean_rr = (pre_rr + post_rr) / 2.0
        
        if peak_idx - win_energy >= 0 and peak_idx + win_energy < len(ecg_filtered):
            r_amp      = ecg_filtered[peak_idx]
            qrs_energy = np.sum(ecg_filtered[peak_idx - win_energy : peak_idx + win_energy] ** 2)
            qrs_width  = get_qrs_width(ecg_filtered, peak_idx, win_energy) / fs * 1000
            
            vec = np.array([[pre_rr, post_rr, mean_rr, r_amp, qrs_energy, qrs_width]])
            pred = clf_results["clf"].predict(clf_results["scaler"].transform(vec))[0]
            beat_preds.append(pred)

    pvc_count  = beat_preds.count(1)
    pac_count  = beat_preds.count(2)
    brad_count = beat_preds.count(3)
    total_eval = len(beat_preds) if len(beat_preds) > 0 else 1

    if pvc_count >= 3:
        target_pred = 1
        conf = pvc_count / total_eval
    elif pac_count >= 3:
        target_pred = 2
        conf = pac_count / total_eval
    elif brad_count > (total_eval * 0.5):
        target_pred = 3
        conf = brad_count / total_eval
    else:
        target_pred = 0
        conf = beat_preds.count(0) / total_eval

    diag_name, diag_color = DISEASE_UI[target_pred]
    print(f"\n    >> AI Diagnosis : {diag_name}")
    print(f"       Abnormal Beats : PVC({pvc_count}), PAC({pac_count})")
    print(f"       Class index  : {target_pred} ({clf_results['target_names'][target_pred]})")

    clf_results["pred_class"]  = int(target_pred)
    clf_results["prediction"]  = diag_name
    clf_results["confidence"]  = conf

    visualize_all(ecg_raw, ecg_filtered, r_peaks, features, freq_results, clf_results, fs)
    plot_feature_importance(clf_results)
    triple_live_comparison_monitor(data_dir=data_dir, fs=fs)
    live_scrolling_monitor_advanced(target_record, clf_results, data_dir=data_dir, fs=fs)

    print("\n[Done] All steps complete.")


if __name__ == "__main__":
    run_pipeline(target_record="200", data_dir="mitdb_data")