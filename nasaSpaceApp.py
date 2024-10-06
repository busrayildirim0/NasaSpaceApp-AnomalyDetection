import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pywt
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from datetime import datetime

# Veri setini yükleyin
cat_directory = "./data/lunar/training/catalogs/"
cat_file = cat_directory + "apollo12_catalog_GradeA_final.csv"
cat = pd.read_csv(cat_file)

# Hedef değişken (velocity) ve özellik (time_rel) belirleyin
row = cat.iloc[6]
arrival_time = datetime.strptime(
    row["time_abs(%Y-%m-%dT%H:%M:%S.%f)"], "%Y-%m-%dT%H:%M:%S.%f"
)
arrival_time_rel = row["time_rel(sec)"]
test_filename = row.filename

data_directory = "./data/lunar/training/data/S12_GradeA/"
csv_file = f"{data_directory}{test_filename}.csv"
data_cat = pd.read_csv(csv_file)

# 500 bin veriden 10 bin veri seçin
data = data_cat.sample(n=100, random_state=42)
print(data)

# 'time_rel(sec)' sütunu seçiliyor
signal = data["time_rel(sec)"].values

# Zaman vektörü, 'signal' üzerinde doğru olmalıdır
N = len(signal)
T = 0.01  # Sabit zaman aralığı (örnekleme frekansı varsayıldı)
yf = fft(signal)
xf = fftfreq(N, T)[: N // 2]

# Fourier dönüşüm grafiği
plt.plot(xf, 2.0 / N * np.abs(yf[: N // 2]))
plt.title("Fourier Dönüşümü")
plt.xlabel("Frekans (Hz)")
plt.ylabel("Genlik")
plt.grid()
plt.show()

# Zaman serisi ve Wavelet ile gürültü azaltma
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

# Zirve noktaları bulma (peaks)
peaks, _ = find_peaks(denoised_signal, height=0.5, distance=5)

plt.figure(figsize=(12, 6))
plt.plot(time, denoised_signal, label="Denoised Signal", color="red")
plt.plot(time[peaks], denoised_signal[peaks], "x", label="Detected Peaks", color="blue")
plt.title("Seismic Event Detection")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

print(f"Number of detected seismic events: {len(peaks)}")
print(f"Seismic event timestamps (indices): {peaks}")


# Yardımcı fonksiyonlar
def load_seismic_data(file_path):
    data = pd.read_csv(file_path)
    return data


def apply_fft(signal, sample_spacing):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, sample_spacing)[: N // 2]
    return xf, 2.0 / N * np.abs(yf[: N // 2])


def wavelet_denoising(signal, wavelet="db4", level=6):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.std(coeffs[-1])
    denoised_coeffs = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_signal


def detect_seismic_events(denoised_signal, height_threshold=0.1, distance=1):
    peaks, _ = find_peaks(denoised_signal, height=height_threshold, distance=distance)
    return peaks


def analyze_seismic_data(file_path, sample_spacing=0.01):
    data = load_seismic_data(file_path)
    print(data.head())

    signal = data["time_rel(sec)"].values

    xf, yf = apply_fft(signal, sample_spacing)

    plt.plot(xf, yf)
    plt.title("Frequency domain (FFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    denoised_signal = wavelet_denoising(signal)

    peaks = detect_seismic_events(denoised_signal)

    time = np.arange(len(signal))
    plt.plot(time, denoised_signal, label="Denoised Signal")
    plt.plot(time[peaks], denoised_signal[peaks], "x", label="Detected Peaks")
    plt.title("Seismic Event Detection")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    print(f"Detected seismic event timestamps: {peaks}")


data["time_abs(%Y-%m-%dT%H:%M:%S.%f)"] = pd.to_datetime(
    data["time_abs(%Y-%m-%dT%H:%M:%S.%f)"]
)
data["time_abs(epoch)"] = data["time_abs(%Y-%m-%dT%H:%M:%S.%f)"].apply(
    lambda x: x.timestamp()
)

# Özellikler belirleme
data["time_abs(%Y-%m-%dT%H:%M:%S.%f)"] = pd.to_datetime(
    data["time_abs(%Y-%m-%dT%H:%M:%S.%f)"]
)
data["time_abs(epoch)"] = data["time_abs(%Y-%m-%dT%H:%M:%S.%f)"].apply(
    lambda x: x.timestamp()
)

# Özellikler belirleme
features = ["time_abs(epoch)", "time_rel(sec)", "velocity(m/s)"]
X = data[features]

# Standartlaştırma
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest modeli oluşturma ve eğitme
contamination = 0.5
model = IsolationForest(contamination=contamination, n_estimators=100, random_state=42)
model.fit(X_scaled)

# Anomali skorlarını hesaplama
anomaly_scores = model.decision_function(X_scaled)
anomaly_labels = model.predict(X_scaled)

# Anomali etiketlerini veri çerçevesine ekleme
data["anomaly_score"] = anomaly_scores
data["is_anomaly"] = anomaly_labels
print(data["anomaly_score"])

# Anomali skoru grafiği oluşturma
plt.figure(figsize=(12, 6))

# Her bir veri noktasının x ekseninde indexini kullanarak grafiği oluşturuyoruz
plt.scatter(
    data.index, data["anomaly_score"], c=data["is_anomaly"], cmap="coolwarm", alpha=0.6
)
plt.colorbar(label="Anomaly")
plt.title("Anomaly Scores in Seismic Data")
plt.xlabel("Index")
plt.ylabel("Anomaly Score")
plt.axhline(y=0, color="red", linestyle="--", label="Anomaly Score Threshold")
plt.legend()
plt.grid()
plt.show()

anomali_points = data[data["is_anomaly"] == 1]
anomali_points = anomali_points.sort_values(by="anomaly_score")
print(len(anomali_points))
# Anomalileri tabloya dönüştür ve yazı boyutlarını ayarla
fig, ax = plt.subplots(figsize=(10, 4))  # Tablo boyutunu artır
ax.axis("tight")
ax.axis("off")

for i in anomali_points:
    data = data.sort_values(by="anomaly_score", ascending=True)
    print(data["anomaly_score"])


# Tabloyu oluşturS
table = ax.table(
    cellText=anomali_points.values,
    colLabels=anomali_points.columns,
    cellLoc="center",
    loc="center",
    bbox=[0.2, 0.2, 0.6, 0.6],  # Tabloyu çerçeve içerisine al
)

# Yazı tipini ve hücre boyutlarını ayarla
table.auto_set_font_size(False)
table.set_fontsize(10)  # Yazı boyutu
table.scale(1.5, 1.5)  # Tabloyu genişlet

# Sütun genişliklerini otomatik ayarla
table.auto_set_column_width(col=list(range(len(anomali_points.columns))))

# Grafik öğelerinin düzgün yerleşmesi için 'tight_layout' kullan
plt.title("Detected Anomalies", fontsize=12)

# Arka planda eksenleri kapat, sadece tabloyu göster
ax.axis("off")

# Düzgün yerleşim için tight_layout kullan
plt.tight_layout()

# Tabloyu göster
plt.show()
