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
from tkinter import messagebox

cat_directory = "./data/lunar/training/catalogs/"
cat_file = f"{cat_directory}apollo12_catalog_GradeA_final.csv"
cat = pd.read_csv(cat_file)


def analyze_seismic_data(input_years, sample_size):
    while True:
        try:
            current_year = datetime.now().year
            filtered_cat = cat[
                cat["time_abs(%Y-%m-%dT%H:%M:%S.%f)"].apply(
                    lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f").year
                    >= (current_year - input_years)
                )
            ]

            if filtered_cat.empty:
                messagebox.showerror(
                    "Error",
                    f"No data found for {input_years} years back. Please enter a different year.",
                )
                return
            else:
                print(f"Filtered data count: {filtered_cat.shape[0]}")
                break

        except ValueError:
            messagebox.showerror(
                "Error", "You entered an invalid value. Please enter a number."
            )
            return

    row = filtered_cat.iloc[0]
    arrival_time = pd.to_datetime(row["time_abs(%Y-%m-%dT%H:%M:%S.%f)"])
    arrival_time_rel = row["time_rel(sec)"]
    test_filename = row.filename

    data_directory = "./data/lunar/training/data/S12_GradeA/"
    csv_file = f"{data_directory}{test_filename}.csv"
    data_cat = pd.read_csv(csv_file)

    data = data_cat.sample(n=sample_size, random_state=42)
    print(data)

    signal = data["time_rel(sec)"].values

    N = len(signal)
    T = 0.01
    yf = fft(signal)
    xf = fftfreq(N, T)[: N // 2]

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

    data["time_abs(%Y-%m-%dT%H:%M:%S.%f)"] = pd.to_datetime(
        data["time_abs(%Y-%m-%dT%H:%M:%S.%f)"]
    )
    data["time_abs(epoch)"] = data["time_abs(%Y-%m-%dT%H:%M:%S.%f)"].apply(
        lambda x: x.timestamp()
    )

    features = ["time_abs(epoch)", "time_rel(sec)", "velocity(m/s)"]
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
    try:
        input_years = int(years_entry.get())
        sample_size = int(sample_entry.get())
        analyze_seismic_data(input_years, sample_size)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers.")


app = tk.Tk()
app.title("Seismic Data Analysis")
app.geometry("600x500")
app.configure(bg="#f0f0f0")

header = tk.Label(
    app, text="Seismic Data Analysis", font=("Helvetica", 18, "bold"), bg="#f0f0f0"
)
header.pack(pady=10)

tk.Label(
    app,
    text="How many years back would you like to analyze the data? (recommended 70 years)",
    bg="#f0f0f0",
).pack(pady=5)
years_entry = tk.Entry(app, font=("Helvetica", 14))
years_entry.pack(pady=5)
tk.Label(
    app,
    text="Increasing the sample value prolongs processing time and makes reading the graph more difficult. \nTo read the graph better, it is recommended to use: 1-5 thousand.",
    bg="#f0f0f0",
).pack(pady=5)

tk.Label(
    app,
    text="Sample size (n) can take values between 0-500,000 (recommended 5-1,000)",
    bg="#f0f0f0",
).pack(pady=5)
sample_entry = tk.Entry(app, font=("Helvetica", 14))
sample_entry.pack(pady=5)

analyze_button = tk.Button(
    app, text="Perform Analysis", command=run_analysis, font=("Helvetica", 14)
)
analyze_button.pack(pady=20)

app.mainloop()
