"""
Gera as 7 figuras para publicação no BRACIS.
  - Salvo em reports/plots/ como PNG, 300 DPI, fundo branco
  - Sem título embutido na figura
  - Paleta grayscale-friendly (azul/laranja, sem vermelho+verde juntos)
  - TF importado antes do sklearn (conflito de DLL no Windows)
"""
import os, sys, pickle
import numpy as np

import matplotlib
matplotlib.use("Agg")   # headless, sem display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# TF antes do sklearn
import tensorflow as tf

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT  = os.path.join(PROJ, "reports", "plots")
os.makedirs(OUT, exist_ok=True)

def out(name):
    return os.path.join(OUT, name)

DPI = 300

# ---------------------------------------------------------------------------
# Paleta grayscale-friendly
#   RF group  → azuis  (distinguíveis em cinza pela luminância)
#   SVM group → laranjas/marrons (mesma lógica)
#   Scatter benigno/maligno → azul × laranja (padrão acessível)
# ---------------------------------------------------------------------------
RF_CORES  = ["#c6dbef", "#4393c3", "#1a3f6f"]   # claro → escuro
SVM_CORES = ["#fdd0a2", "#f16913", "#7f2704"]
CORES_6   = RF_CORES + SVM_CORES
C_BEN = "#1f77b4"   # azul
C_MAL = "#ff7f0e"   # laranja

LABELS_6 = ["RF Original", "RF PCA", "RF AE",
             "SVM Original", "SVM PCA", "SVM AE"]

# Estilo base
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "lines.linewidth":   1.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Carrega dados compartilhados
# ---------------------------------------------------------------------------
print("Carregando dados...")

with open(os.path.join(PROJ, "models", "acc_results.pkl"), "rb") as f:
    res = pickle.load(f)

dados_box = [
    res["acc_rf_o"], res["acc_rf_pca"], res["acc_rf_ae"],
    res["acc_svm_o"], res["acc_svm_pca"], res["acc_svm_ae"],
]

with open(os.path.join(PROJ, "models", "ae_history.pkl"), "rb") as f:
    ae_history = pickle.load(f)

encoder = tf.keras.models.load_model(
    os.path.join(PROJ, "models", "encoder.keras")
)

data_bc   = load_breast_cancer()
X_bc      = data_bc.data
y_bc      = data_bc.target
X_scaled  = StandardScaler().fit_transform(X_bc)

print("  acc_results.pkl  OK")
print("  ae_history.pkl   OK")
print("  encoder.keras    OK")
print("  breast_cancer    OK")

# ===========================================================================
# Fig 1 — Boxplot acurácias (6 configurações × 30 repetições)
# ===========================================================================
print("\nFig1 — boxplot...")

fig, ax = plt.subplots(figsize=(9, 5))

bp = ax.boxplot(
    dados_box,
    patch_artist=True,
    notch=False,
    widths=0.55,
    medianprops=dict(color="black", linewidth=2.2),
    whiskerprops=dict(color="#333333", linewidth=1.2),
    capprops=dict(color="#333333", linewidth=1.2),
    flierprops=dict(marker="o", markersize=4, markeredgecolor="#555555",
                    markerfacecolor="none", linestyle="none"),
)
for patch, color in zip(bp["boxes"], CORES_6):
    patch.set_facecolor(color)
    patch.set_alpha(0.80)

ax.set_xticks(range(1, 7))
ax.set_xticklabels(LABELS_6, rotation=18, ha="right")
ax.set_ylabel("Acurácia")
ax.set_ylim(0.88, 1.00)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
ax.grid(axis="y", linestyle="--", alpha=0.45)

# Linha divisória RF × SVM
ax.axvline(3.5, color="#aaaaaa", linestyle=":", linewidth=1.2)
ax.text(2.0, 0.997, "RF", ha="center", fontsize=9, color="#555555")
ax.text(5.0, 0.997, "SVM", ha="center", fontsize=9, color="#555555")

fig.tight_layout()
fig.savefig(out("Fig1_boxplot.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  salva.")

# ===========================================================================
# Fig 2 — Ranking médio de Friedman (barras horizontais)
# ===========================================================================
print("Fig2 — friedman ranking...")

matriz    = np.vstack(dados_box).T          # (30, 6)
rankings  = np.argsort(np.argsort(-matriz, axis=1), axis=1) + 1
rank_med  = rankings.mean(axis=0)

ordem      = np.argsort(rank_med)           # do melhor para o pior
labels_ord = [LABELS_6[i] for i in ordem]
rank_ord   = rank_med[ordem]
cores_ord  = [CORES_6[i] for i in ordem]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.barh(
    range(len(ordem)), rank_ord,
    color=cores_ord, edgecolor="black", linewidth=0.8, alpha=0.85,
)
ax.set_yticks(range(len(ordem)))
ax.set_yticklabels(labels_ord)
ax.set_xlabel("Ranking Médio (menor = melhor)")
ax.set_xlim(0, 6.8)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.invert_yaxis()

for bar, val in zip(bars, rank_ord):
    ax.text(val + 0.08, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=9.5)

ax.grid(axis="x", linestyle="--", alpha=0.45)
fig.tight_layout()
fig.savefig(out("Fig2_friedman_ranking.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  salva.")

# ===========================================================================
# Fig 3 — Heatmap diferenças absolutas de acurácia
# ===========================================================================
print("Fig3 — heatmap...")

medias      = np.array([v.mean() for v in dados_box])
diff_matrix = np.abs(medias.reshape(-1, 1) - medias.reshape(1, -1))

fig, ax = plt.subplots(figsize=(8, 6.2))
sns.heatmap(
    diff_matrix,
    annot=True,
    fmt=".3f",
    cmap="Blues",
    xticklabels=LABELS_6,
    yticklabels=LABELS_6,
    ax=ax,
    linewidths=0.5,
    linecolor="#cccccc",
    annot_kws={"size": 9.5},
    cbar_kws={"shrink": 0.8},
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(out("Fig3_heatmap.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  salva.")

# ===========================================================================
# Fig 4 — PCA 2D scatter
# ===========================================================================
print("Fig4 — PCA 2D scatter...")

pca2d  = PCA(n_components=2)
Z_pca2d = pca2d.fit_transform(X_scaled)

var1 = pca2d.explained_variance_ratio_[0] * 100
var2 = pca2d.explained_variance_ratio_[1] * 100

fig, ax = plt.subplots(figsize=(7, 6))
for label, color, marker, name in [
    (1, C_MAL, "^", "Maligno"),
    (0, C_BEN, "o", "Benigno"),   # benigno por cima para legibilidade
]:
    mask = y_bc == label
    ax.scatter(Z_pca2d[mask, 0], Z_pca2d[mask, 1],
               c=color, marker=marker, s=38, alpha=0.72,
               edgecolors="none", label=name, zorder=3 if label == 0 else 2)

ax.set_xlabel(f"Componente Principal 1 ({var1:.1f}% var.)")
ax.set_ylabel(f"Componente Principal 2 ({var2:.1f}% var.)")
ax.legend(loc="upper right", framealpha=0.9, edgecolor="#cccccc")
ax.grid(True, linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig(out("Fig4_pca2d.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  salva.")

# ===========================================================================
# Fig 5 — Espaço latente do Autoencoder 3D
# ===========================================================================
print("Fig5 — AE 3D scatter...")

Z_ae = encoder.predict(X_scaled, verbose=0)

fig = plt.figure(figsize=(8, 7))
fig.patch.set_facecolor("white")
ax3d = fig.add_subplot(111, projection="3d")
ax3d.set_facecolor("white")

for label, color, marker, name in [
    (1, C_MAL, "^", "Maligno"),
    (0, C_BEN, "o", "Benigno"),
]:
    mask = y_bc == label
    ax3d.scatter(Z_ae[mask, 0], Z_ae[mask, 1], Z_ae[mask, 2],
                 c=color, marker=marker, s=28, alpha=0.75, label=name)

ax3d.set_xlabel("Dim. Latente 1", labelpad=8, fontsize=10)
ax3d.set_ylabel("Dim. Latente 2", labelpad=8, fontsize=10)
ax3d.set_zlabel("Dim. Latente 3", labelpad=8, fontsize=10)
ax3d.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax3d.view_init(elev=20, azim=45)

fig.tight_layout()
fig.savefig(out("Fig5_ae3d.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  salva.")

# ===========================================================================
# Fig 6 — Curva de treinamento do Autoencoder
# ===========================================================================
print("Fig6 — curva AE...")

loss   = ae_history["loss"]
epocas = np.arange(1, len(loss) + 1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(epocas, loss, color="#1a3f6f", linewidth=2)
ax.fill_between(epocas, loss, alpha=0.12, color="#1a3f6f")
ax.set_xlabel("Épocas")
ax.set_ylabel("Loss (MSE)")
ax.set_xlim(1, len(loss))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.grid(True, linestyle="--", alpha=0.45)
fig.tight_layout()
fig.savefig(out("Fig6_curva_ae.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  salva.")

# ===========================================================================
# Fig 7 — Diagrama arquitetura Autoencoder (matplotlib puro)
# ===========================================================================
print("Fig7 — arquitetura AE...")

# Definição das camadas: (label, facecolor, textcolor)
LAYERS = [
    ("Input\n(30)",     "#f0f0f0", "black"),
    ("Dense 16\nReLU",  "#c6dbef", "black"),
    ("Dense 8\nReLU",   "#4393c3", "white"),
    ("Latente\n(3)",    "#1a3f6f", "white"),
    ("Dense 8\nReLU",   "#4393c3", "white"),
    ("Dense 16\nReLU",  "#c6dbef", "black"),
    ("Output\n(30)",    "#f0f0f0", "black"),
]

N      = len(LAYERS)
XS     = np.linspace(1.0, 13.0, N)   # centros horizontais
BOX_W  = 1.55
BOX_H  = 1.0
Y_MID  = 1.6    # centro vertical das caixas
Y_BRACE= 3.0    # y dos rótulos de seção

fig, ax = plt.subplots(figsize=(14, 4.2))
ax.set_xlim(0, 14)
ax.set_ylim(0, 3.8)
ax.axis("off")
fig.patch.set_facecolor("white")

# Caixas
for (label, fc, tc), x in zip(LAYERS, XS):
    rect = plt.Rectangle(
        (x - BOX_W / 2, Y_MID - BOX_H / 2),
        BOX_W, BOX_H,
        facecolor=fc, edgecolor="#333333", linewidth=1.4, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, Y_MID, label,
            ha="center", va="center", color=tc,
            fontsize=9, fontweight="bold", zorder=4)

# Setas entre caixas
for i in range(N - 1):
    ax.annotate(
        "",
        xy=(XS[i + 1] - BOX_W / 2 - 0.06, Y_MID),
        xytext=(XS[i] + BOX_W / 2 + 0.06, Y_MID),
        arrowprops=dict(arrowstyle="-|>", color="#333333",
                        lw=1.6, mutation_scale=14),
        zorder=2,
    )

# Rótulos de seção com chaves/suportes
sections = [
    (0, 0, "Input"),
    (1, 2, "Encoder"),
    (3, 3, "Latente"),
    (4, 5, "Decoder"),
    (6, 6, "Output"),
]
Y_LABEL = Y_MID + BOX_H / 2 + 0.55

for s, e, sec in sections:
    x_s = XS[s] - BOX_W / 2 + 0.08
    x_e = XS[e] + BOX_W / 2 - 0.08
    x_mid = (XS[s] + XS[e]) / 2

    # linha horizontal
    ax.plot([x_s, x_e], [Y_LABEL - 0.15, Y_LABEL - 0.15],
            color="#555555", linewidth=1.0)
    # ticks verticais nas pontas
    ax.plot([x_s, x_s], [Y_LABEL - 0.25, Y_LABEL - 0.15], color="#555555", lw=1.0)
    ax.plot([x_e, x_e], [Y_LABEL - 0.25, Y_LABEL - 0.15], color="#555555", lw=1.0)
    # rótulo
    ax.text(x_mid, Y_LABEL + 0.05, sec,
            ha="center", va="bottom", fontsize=9.5, color="#222222",
            fontweight="bold" if sec == "Latente" else "normal")

fig.tight_layout()
fig.savefig(out("Fig7_arquitetura_ae.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("  salva.")

# ===========================================================================
# Verificação final
# ===========================================================================
print("\n" + "=" * 60)
print(f"{'Arquivo':<30} {'Tamanho':>10}   DPI")
print("=" * 60)

try:
    from PIL import Image
    use_pil = True
except ImportError:
    use_pil = False

for fname in sorted(os.listdir(OUT)):
    if not fname.endswith(".png"):
        continue
    path = os.path.join(OUT, fname)
    kb   = os.path.getsize(path) / 1024
    if use_pil:
        with Image.open(path) as img:
            dpi_val = img.info.get("dpi", ("?", "?"))
        dpi_str = f"{dpi_val[0]:.0f}×{dpi_val[1]:.0f}" if isinstance(dpi_val[0], (int, float)) else str(dpi_val)
    else:
        dpi_str = f"{DPI} (set on save)"
    print(f"  {fname:<28} {kb:7.1f} KB   {dpi_str}")

print("=" * 60)
print("Concluído.")
