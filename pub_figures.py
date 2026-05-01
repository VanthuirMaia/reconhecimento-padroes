"""Gera as 7 figuras de publicacao com estilo correto (sem titulos, acentos, 300 DPI)."""
import os, pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# TF antes do sklearn para evitar conflito de DLL no Windows
import tensorflow as tf

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Config global ─────────────────────────────────────────────────────────────
plt.rcParams['font.family']        = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

DPI = 300
OUT = 'reports/plots'
os.makedirs(OUT, exist_ok=True)

# ── Dados ─────────────────────────────────────────────────────────────────────
data     = load_breast_cancer()
X_raw, y = data.data, data.target
X_scaled = StandardScaler().fit_transform(X_raw)

with open('models/acc_results.pkl', 'rb') as f:
    res = pickle.load(f)

acc_rf_o    = res['acc_rf_o']
acc_rf_pca  = res['acc_rf_pca']
acc_rf_ae   = res['acc_rf_ae']
acc_svm_o   = res['acc_svm_o']
acc_svm_pca = res['acc_svm_pca']
acc_svm_ae  = res['acc_svm_ae']

with open('models/ae_history.pkl', 'rb') as f:
    ae_history = pickle.load(f)

encoder = tf.keras.models.load_model('models/encoder.keras')

labels6   = ['RF Original', 'RF PCA', 'RF AE', 'SVM Original', 'SVM PCA', 'SVM AE']
dados_box = [acc_rf_o, acc_rf_pca, acc_rf_ae, acc_svm_o, acc_svm_pca, acc_svm_ae]

# ── Fig1: Boxplot ─────────────────────────────────────────────────────────────
sns.set(style='whitegrid', context='talk', palette='deep')
plt.rcParams['font.family']        = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 8))
palette = sns.color_palette('deep', 6)
sns.boxplot(data=dados_box, ax=ax, palette=palette)
ax.set_xticks(range(6))
ax.set_xticklabels(labels6, rotation=30, ha='right')
ax.set_ylabel('Acurácia')
ax.set_xlabel('')
ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
fig.savefig(f'{OUT}/Fig1_boxplot.png', dpi=DPI, facecolor='white')
plt.close(fig)
print('Fig1 salva')

# ── Fig2: Ranking Médio Friedman (ordenado melhor→pior) ───────────────────────
matriz    = np.vstack(dados_box).T
rank_med  = (np.argsort(np.argsort(-matriz, axis=1), axis=1) + 1).mean(axis=0)
order         = np.argsort(rank_med)
labels_sorted = [labels6[i] for i in order]
ranks_sorted  = rank_med[order]
colors_sorted = [palette[i] for i in order]

fig, ax = plt.subplots(figsize=(14, 6))
x    = np.arange(len(labels_sorted))
bars = ax.bar(x, ranks_sorted, color=colors_sorted)
for bar, rk in zip(bars, ranks_sorted):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.07,
            f'{rk:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels_sorted, rotation=15, ha='right')
ax.set_ylabel('Ranking Médio (menor = melhor)')
ax.set_xlabel('')
ax.set_ylim(0, ranks_sorted.max() + 0.9)
ax.grid(axis='y', linestyle='--', alpha=0.5)
fig.tight_layout()
fig.savefig(f'{OUT}/Fig2_friedman_ranking.png', dpi=DPI, facecolor='white')
plt.close(fig)
print('Fig2 salva')

# ── Fig3: Heatmap diferenças absolutas (Blues) ────────────────────────────────
medias      = np.array([a.mean() for a in dados_box])
diff_matrix = np.abs(medias.reshape(-1, 1) - medias.reshape(1, -1))

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(diff_matrix, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=labels6, yticklabels=labels6, ax=ax,
            linewidths=0.5, linecolor='white')
ax.set_xlabel('')
ax.set_ylabel('')
fig.tight_layout()
fig.savefig(f'{OUT}/Fig3_heatmap.png', dpi=DPI, facecolor='white')
plt.close(fig)
print('Fig3 salva')

# ── Fig4: PCA 2D ──────────────────────────────────────────────────────────────
pca_2d   = PCA(n_components=2, random_state=42)
Z_pca_2d = pca_2d.fit_transform(X_scaled)
var1     = pca_2d.explained_variance_ratio_[0] * 100
var2     = pca_2d.explained_variance_ratio_[1] * 100

fig, ax = plt.subplots(figsize=(10, 8))
mask_b = y == 1   # Benigno (classe 1 no WDBC)
mask_m = y == 0   # Maligno (classe 0)

ax.scatter(Z_pca_2d[mask_b, 0], Z_pca_2d[mask_b, 1],
           marker='o', color='#1f77b4', s=40, alpha=0.6, label='Benigno')
ax.scatter(Z_pca_2d[mask_m, 0], Z_pca_2d[mask_m, 1],
           marker='^', color='#ff7f0e', s=40, alpha=0.6, label='Maligno')

ax.set_xlabel(f'Componente Principal 1 ({var1:.1f}% var.)')
ax.set_ylabel(f'Componente Principal 2 ({var2:.1f}% var.)')
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig(f'{OUT}/Fig4_pca2d.png', dpi=DPI, facecolor='white')
plt.close(fig)
print('Fig4 salva')

# ── Fig5: AE 3D ───────────────────────────────────────────────────────────────
Z_ae   = encoder.predict(X_scaled, verbose=0)
mask_b = y == 1
mask_m = y == 0

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig  = plt.figure(figsize=(10, 8))
ax3d = fig.add_subplot(111, projection='3d')

ax3d.scatter(Z_ae[mask_b, 0], Z_ae[mask_b, 1], Z_ae[mask_b, 2],
             marker='o', color='#1f77b4', s=40, alpha=0.6, label='Benigno')
ax3d.scatter(Z_ae[mask_m, 0], Z_ae[mask_m, 1], Z_ae[mask_m, 2],
             marker='^', color='#ff7f0e', s=40, alpha=0.6, label='Maligno')

ax3d.view_init(elev=20, azim=135)
ax3d.set_xlabel('Dim. Latente 1')
ax3d.set_ylabel('Dim. Latente 2')
ax3d.set_zlabel('Dim. Latente 3')
ax3d.legend(loc='upper left')
fig.tight_layout()
fig.savefig(f'{OUT}/Fig5_ae3d.png', dpi=DPI, facecolor='white')
plt.close(fig)
print('Fig5 salva')

# ── Fig6: Curva de treinamento AE ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ae_history['loss'], color='#1f77b4', linewidth=2)
ax.set_xlabel('Épocas')
ax.set_ylabel('Loss (MSE)')
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
fig.savefig(f'{OUT}/Fig6_curva_ae.png', dpi=DPI, facecolor='white')
plt.close(fig)
print('Fig6 salva')

# ── Fig7: Arquitetura do AE ───────────────────────────────────────────────────
layer_labels = ['Input\n(30)', 'Dense 16\nReLU', 'Dense 8\nReLU',
                'Latente\n(3)', 'Dense 8\nReLU', 'Dense 16\nReLU', 'Output\n(30)']

# Cores: Input/Output cinza claro, Encoder gradiente azul claro→médio,
#        Latente azul escuro, Decoder médio→claro
layer_colors = ['#d9d9d9',   # Input
                '#c6dbef',   # Dense 16 (Encoder, mais externo)
                '#6baed6',   # Dense 8  (Encoder)
                '#2171b5',   # Latente  (mais escuro)
                '#6baed6',   # Dense 8  (Decoder)
                '#c6dbef',   # Dense 16 (Decoder, mais externo)
                '#d9d9d9']   # Output

centers  = [(i * 2 + 1, 1.0) for i in range(7)]   # x = 1,3,5,7,9,11,13
BOX_W, BOX_H = 1.5, 0.85

fig, ax = plt.subplots(figsize=(16, 4.5))
ax.set_xlim(0, 14)
ax.set_ylim(-0.2, 3.0)
ax.axis('off')

# Desenhar caixas
for (cx, cy), lbl, color in zip(centers, layer_labels, layer_colors):
    rect = Rectangle((cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
                     fill=True, edgecolor='#333333', facecolor=color,
                     linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, lbl, ha='center', va='center', fontsize=9,
            fontweight='bold', zorder=4)

# Setas entre caixas
for i in range(len(centers) - 1):
    x1 = centers[i][0]   + BOX_W / 2
    x2 = centers[i+1][0] - BOX_W / 2
    y  = centers[0][1]
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.8),
                zorder=5)

# Chaves / agrupamentos no topo
# Grupos: Input=[0], Encoder=[1,2], Latente=[3], Decoder=[4,5], Output=[6]
groups = [
    ('Input',   [0]),
    ('Encoder', [1, 2]),
    ('Latente', [3]),
    ('Decoder', [4, 5]),
    ('Output',  [6]),
]
BRACE_Y   = 1.75   # altura da linha de agrupamento
LABEL_Y   = 2.05   # altura do texto
TICK_H    = 0.12   # altura dos tiques verticais

for name, idxs in groups:
    xs     = [centers[i][0] for i in idxs]
    x_left  = min(xs) - BOX_W / 2 + 0.1
    x_right = max(xs) + BOX_W / 2 - 0.1
    x_mid   = (x_left + x_right) / 2

    # Linha horizontal
    ax.plot([x_left, x_right], [BRACE_Y, BRACE_Y],
            color='#444444', linewidth=1.2)
    # Tiques verticais nas extremidades
    ax.plot([x_left,  x_left],  [BRACE_Y - TICK_H, BRACE_Y], color='#444444', lw=1.2)
    ax.plot([x_right, x_right], [BRACE_Y - TICK_H, BRACE_Y], color='#444444', lw=1.2)
    # Label
    ax.text(x_mid, LABEL_Y, name, ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#222222')

fig.tight_layout()
fig.savefig(f'{OUT}/Fig7_arquitetura_ae.png', dpi=DPI, facecolor='white')
plt.close(fig)
print('Fig7 salva')

print(f'\nTodas as 7 figuras regeneradas em {os.path.abspath(OUT)}/')
