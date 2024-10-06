import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from tabulate import tabulate
import pywt
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, filedialog


def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load the file: {str(e)}")
        return None


def determine_sample_size(data_length):
    if data_length <= 1000:
        return data_length  # Use full dataset if it's small
    elif data_length <= 10000:
        return 1000
    elif data_length <= 100000:
        return 5000
    else:
        return 10000  # Cap at 10000 for very large datasets


def analyze_seismic_data(data):
    if data.empty:
        messagebox.showerror("Error", "The loaded dataset is empty.")
        return

    sample_size = determine_sample_size(len(data))
    data = data.sample(n=sample_size, random_state=42)
    print(f"Analyzing with sample size: {sample_size}")
    print(data)

    signal = data["time_rel(sec)"].values

    N = len(signal)
    T = 0.01
    yf = fft(signal)
    xf = fftfreq(N, T)[: N // 2]

    plt.figure(figsize=(10, 6))
    plt.plot(xf, 2.0 / N * np.abs(yf[: N // 2]))
    plt.title("Fourier Transform")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    time = np.arange(len(signal))
    wavelet = "db4"
    coeffs = pywt.wavedec(signal, wavelet, level=6)
    threshold = np.std(coeffs[-1])
    denoised_coeffs = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

    plt.figure(figsize=(10, 6))
    plt.plot(time, signal, label="Noisy Signal")
    plt.plot(time, denoised_signal, label="Denoised Signal", color="red")
    plt.title("Wavelet Denoising")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    peaks, _ = find_peaks(denoised_signal, height=0.5, distance=30)

    plt.figure(figsize=(12, 6))
    plt.plot(time, denoised_signal, label="Denoised Signal", color="red")
    plt.plot(
        time[peaks], denoised_signal[peaks], "x", label="Detected Peaks", color="blue"
    )
    plt.title("Seismic Event Detection")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()

    if "time_abs(%Y-%m-%dT%H:%M:%S.%f)" in data.columns:
        data["time_abs(%Y-%m-%dT%H:%M:%S.%f)"] = pd.to_datetime(
            data["time_abs(%Y-%m-%dT%H:%M:%S.%f)"]
        )
        data["time_abs(epoch)"] = data["time_abs(%Y-%m-%dT%H:%M:%S.%f)"].apply(
            lambda x: x.timestamp()
        )
        features = ["time_abs(epoch)", "time_rel(sec)", "velocity(m/s)"]
    else:
        features = ["time_rel(sec)", "velocity(m/s)"]

    X = data[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.5, n_estimators=80, random_state=42)
    model.fit(X_scaled)

    anomaly_scores = model.decision_function(X_scaled)
    anomaly_labels = model.predict(X_scaled)

    data["anomaly_score"] = anomaly_scores
    data["is_anomaly"] = anomaly_labels

    plt.figure(figsize=(12, 6))
    plt.scatter(
        data.index,
        data["anomaly_score"],
        c=data["is_anomaly"],
        cmap="coolwarm",
        alpha=0.6,
    )
    plt.colorbar(label="Anomaly")
    plt.title("Anomaly Scores in Seismic Data")
    plt.xlabel("Index")
    plt.ylabel("Anomaly Score")
    plt.axhline(y=0, color="red", linestyle="--", label="Anomaly Score Threshold")
    plt.legend()
    plt.grid()
    plt.show()

    anomaly_points = data[
        (data["anomaly_score"] >= 0.027)
        & (data["anomaly_score"] <= 0.05)
        & (data["is_anomaly"] == 1)
    ].sort_values(by="anomaly_score")

    if not anomaly_points.empty:
        print("Detected seismic events (score 0.027 - 0.05):")
        print(tabulate(anomaly_points, headers="keys", tablefmt="grid"))
    else:
        print("No seismic events found.")

    print(f"Number of detected seismic events: {len(anomaly_points)}")


def run_analysis():
    global data  # Make data a global variable
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return

    data = load_data(file_path)
    if data is None:
        return

    messagebox.showinfo(
        "Info",
        f"Data loaded successfully. Total rows: {len(data)}. Click 'Analyze' to process the data.",
    )
    analyze_button.config(state=tk.NORMAL)  # Enable the Analyze button


def perform_analysis():
    try:
        analyze_seismic_data(data)
    except NameError:
        messagebox.showerror("Error", "Please load data first before analyzing.")


app = tk.Tk()
app.title("Seismic Data Analysis")
app.geometry("400x200")
app.configure(bg="#f0f0f0")

header = tk.Label(
    app, text="Seismic Data Analysis", font=("Helvetica", 18, "bold"), bg="#f0f0f0"
)
header.pack(pady=10)

load_button = tk.Button(
    app, text="Load Data", command=run_analysis, font=("Helvetica", 14)
)
load_button.pack(pady=10)

analyze_button = tk.Button(
    app,
    text="Analyze",
    command=perform_analysis,
    font=("Helvetica", 14),
    state=tk.DISABLED,
)
analyze_button.pack(pady=10)

app.mainloop()
