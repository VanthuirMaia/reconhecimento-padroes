"""Regenera as 7 figuras de publicacao com random_state=42."""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

sys.path.insert(0, os.path.abspath('.'))
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SEED = 42
OUT  = 'reports/plots'
os.makedirs(OUT, exist_ok=True)
DPI  = 300

sns.set(style="whitegrid", context="talk", palette="deep")

# ── Carregar dados ─────────────────────────────────────────────────────────────
with open('models/acc_results.pkl', 'rb') as f:
    resultados = pickle.load(f)

acc_rf_o   = resultados['acc_rf_o']
acc_rf_pca = resultados['acc_rf_pca']
acc_rf_ae  = resultados['acc_rf_ae']
acc_svm_o  = resultados['acc_svm_o']
acc_svm_pca= resultados['acc_svm_pca']
acc_svm_ae = resultados['acc_svm_ae']

with open('models/ae_history.pkl', 'rb') as f:
    ae_history = pickle.load(f)

encoder = tf.keras.models.load_model('models/encoder.keras')

data    = load_breast_cancer()
X_raw   = data.data
y       = data.target
scaler  = StandardScaler()
X_scaled= scaler.fit_transform(X_raw)

labels6 = ["RF Original", "RF PCA", "RF AE", "SVM Original", "SVM PCA", "SVM AE"]
dados_box = [acc_rf_o, acc_rf_pca, acc_rf_ae, acc_svm_o, acc_svm_pca, acc_svm_ae]

# ── Fig1: Boxplot (sem rotulos flutuantes) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 8))
sns.boxplot(data=dados_box, ax=ax,
            palette=sns.color_palette("deep", 6))
ax.set_xticks(range(6))
ax.set_xticklabels(labels6, rotation=15)
ax.set_title("Comparacao de Acuracia (30 execucoes)", fontsize=14)
ax.set_ylabel("Acuracia")
ax.set_xlabel("Modelos")
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(f'{OUT}/Fig1_boxplot.png', dpi=DPI)
plt.close(fig)
print("Fig1 salva")

# ── Fig2: Ranking Medio Friedman ───────────────────────────────────────────────
matriz = np.vstack([acc_rf_o, acc_rf_pca, acc_rf_ae,
                    acc_svm_o, acc_svm_pca, acc_svm_ae]).T
rankings = np.argsort(np.argsort(-matriz, axis=1), axis=1) + 1
ranking_medio = rankings.mean(axis=0)

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(labels6, ranking_medio, color=sns.color_palette("deep", 6))
ax.set_title("Ranking Medio dos Modelos (Friedman)", fontsize=14)
ax.set_ylabel("Ranking Medio (menor e melhor)")
ax.set_xticklabels(labels6, rotation=15)
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
fig.savefig(f'{OUT}/Fig2_friedman_ranking.png', dpi=DPI)
plt.close(fig)
print("Fig2 salva")

# ── Fig3: Heatmap de diferencas absolutas ─────────────────────────────────────
medias = np.array([a.mean() for a in dados_box])
diff_matrix = np.abs(medias.reshape(-1, 1) - medias.reshape(1, -1))

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(diff_matrix, annot=True, fmt=".3f", cmap="coolwarm",
            xticklabels=labels6, yticklabels=labels6, ax=ax)
ax.set_title("Heatmap das Diferencas Absolutas de Acuracia Entre os Modelos")
fig.tight_layout()
fig.savefig(f'{OUT}/Fig3_heatmap.png', dpi=DPI)
plt.close(fig)
print("Fig3 salva")

# ── Fig4: PCA 2D (s=40, alpha=0.6) ────────────────────────────────────────────
pca_2d = PCA(n_components=2, random_state=SEED)
Z_pca_2d = pca_2d.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 8))
scatter_ax = sns.scatterplot(
    x=Z_pca_2d[:, 0], y=Z_pca_2d[:, 1],
    hue=y,
    palette=["#1f77b4", "#ff7f0e"],
    s=40, alpha=0.6, ax=ax
)
for t, lbl in zip(scatter_ax.legend_.texts, ["Benigno", "Maligno"]):
    t.set_text(lbl)
ax.set_title("Representacao PCA em 2D -- Dataset WDBC")
ax.set_xlabel("Componente Principal 1")
ax.set_ylabel("Componente Principal 2")
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(f'{OUT}/Fig4_pca2d.png', dpi=DPI)
plt.close(fig)
print("Fig4 salva")

# ── Fig5: AE 3D (azimuth=135, elevation=20) ───────────────────────────────────
Z_ae = encoder.predict(X_scaled, verbose=0)

fig = plt.figure(figsize=(10, 8))
ax3d = fig.add_subplot(111, projection='3d')
sc = ax3d.scatter(Z_ae[:, 0], Z_ae[:, 1], Z_ae[:, 2],
                  c=y, cmap="coolwarm", s=40, alpha=0.6)
ax3d.view_init(elev=20, azim=135)
ax3d.set_title("Representacao 3D do Espaco Latente -- Autoencoder")
ax3d.set_xlabel("Dimensao Latente 1")
ax3d.set_ylabel("Dimensao Latente 2")
ax3d.set_zlabel("Dimensao Latente 3")
fig.colorbar(sc, label="Classe (0=Benigno, 1=Maligno)", shrink=0.6)
fig.tight_layout()
fig.savefig(f'{OUT}/Fig5_ae3d.png', dpi=DPI)
plt.close(fig)
print("Fig5 salva")

# ── Fig6: Curva de treino do autoencoder ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ae_history["loss"], label="Loss Treino", linewidth=2)
if "val_loss" in ae_history:
    ax.plot(ae_history["val_loss"], label="Loss Validacao", linewidth=2)
ax.set_title("Curva de Treinamento do Autoencoder")
ax.set_xlabel("Epocas")
ax.set_ylabel("Loss (MSE)")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(f'{OUT}/Fig6_curva_ae.png', dpi=DPI)
plt.close(fig)
print("Fig6 salva")

# ── Fig7: Arquitetura do autoencoder ──────────────────────────────────────────
from matplotlib.patches import Rectangle

def draw_layer(ax, center, width, height, label):
    x, y_ = center[0] - width/2, center[1] - height/2
    rect = Rectangle((x, y_), width, height, fill=True,
                     edgecolor='black', facecolor='#dce6f2', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(center[0], center[1], label, ha='center', va='center', fontsize=11)

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
centers = [(1,1),(3,1),(5,1),(7,1),(9,1),(11,1),(13,1)]
labels7 = ["Input\n(30)","Dense 16\nReLU","Dense 8\nReLU",
           "Latente (3)","Dense 8\nReLU","Dense 16\nReLU","Output\n(30)"]
for c, lbl in zip(centers, labels7):
    draw_layer(ax, c, width=1.4, height=0.9, label=lbl)
for i in range(len(centers)-1):
    x1, y1 = centers[i]; x2, y2 = centers[i+1]
    ax.annotate("", xy=(x2-0.9, y2), xytext=(x1+0.9, y1),
                arrowprops=dict(arrowstyle="->", lw=2))
ax.set_xlim(0, 14); ax.set_ylim(0, 2)
ax.set_title("Arquitetura do Autoencoder Utilizado no Sistema Hibrido", fontsize=14)
fig.tight_layout()
fig.savefig(f'{OUT}/Fig7_arquitetura_ae.png', dpi=DPI)
plt.close(fig)
print("Fig7 salva")

print(f"\nTodas as 7 figuras salvas em {OUT}/")
