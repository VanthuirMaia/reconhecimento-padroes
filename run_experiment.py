import os, sys, pickle, joblib
os.environ['PYTHONHASHSEED'] = '42'
import random; random.seed(42)
import numpy as np; np.random.seed(42)
import tensorflow as tf; tf.random.set_seed(42)

sys.path.insert(0, os.path.abspath('.'))
SEED = 42

# ─── NB01 ─────────────────────────────────────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X_all, y_all = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)
scaler_nb01 = StandardScaler()
X_train_scaled = scaler_nb01.fit_transform(X_train)
X_test_scaled  = scaler_nb01.transform(X_test)

os.makedirs('data/processed', exist_ok=True)
joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test),
            'data/processed/scaled_data.pkl')
print(f"[NB01] scaled_data.pkl salvo — treino: {X_train_scaled.shape}")

# ─── NB04 — Autoencoder ───────────────────────────────────────────────────────
np.random.seed(SEED)
tf.random.set_seed(SEED)

from src.autoencoder import create_autoencoder, train_autoencoder

autoencoder, encoder = create_autoencoder(input_dim=30, bottleneck_dim=3, seed=SEED)
history = train_autoencoder(autoencoder, X_train_scaled, epochs=40)

os.makedirs('models', exist_ok=True)
encoder.save('models/encoder.keras')
with open('models/ae_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

final_loss = history.history['loss'][-1]
print(f"[NB04] encoder.keras + ae_history.pkl salvos")
print(f"[NB04] Loss final (epoca 40): {final_loss:.6f}")

# ─── NB05 — 30 repeticoes ─────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare

X, y = X_all, y_all

def avaliar_repetidamente(modelo, X, y, reps=30, pca=None, ae=None):
    acc_l, prec_l, rec_l, f1_l = [], [], [], []
    for seed in range(reps):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        if pca is not None:
            Ztr = pca.fit_transform(Xtr_s)
            Zte = pca.transform(Xte_s)
        elif ae is not None:
            Ztr = ae.predict(Xtr_s, verbose=0)
            Zte = ae.predict(Xte_s, verbose=0)
        else:
            Ztr, Zte = Xtr_s, Xte_s
        modelo.fit(Ztr, ytr)
        yp = modelo.predict(Zte)
        acc_l.append(accuracy_score(yte, yp))
        prec_l.append(precision_score(yte, yp, average='weighted', zero_division=0))
        rec_l.append(recall_score(yte, yp, average='weighted', zero_division=0))
        f1_l.append(f1_score(yte, yp, average='weighted', zero_division=0))
    return {
        'acc':  np.array(acc_l),
        'prec': np.array(prec_l),
        'rec':  np.array(rec_l),
        'f1':   np.array(f1_l),
    }

rf  = RandomForestClassifier(random_state=SEED)
svm = SVC(random_state=SEED)
pca = PCA(n_components=7, random_state=SEED)

print("[NB05] Rodando 30 repeticoes...")
res_rf_o   = avaliar_repetidamente(rf,  X, y); print("  RF_original OK")
res_rf_pca = avaliar_repetidamente(rf,  X, y, pca=pca); print("  RF_PCA OK")
res_rf_ae  = avaliar_repetidamente(rf,  X, y, ae=encoder); print("  RF_AE OK")
res_svm_o  = avaliar_repetidamente(svm, X, y); print("  SVM_original OK")
res_svm_pca= avaliar_repetidamente(svm, X, y, pca=pca); print("  SVM_PCA OK")
res_svm_ae = avaliar_repetidamente(svm, X, y, ae=encoder); print("  SVM_AE OK")

acc_rf_o    = res_rf_o['acc']
acc_rf_pca  = res_rf_pca['acc']
acc_rf_ae   = res_rf_ae['acc']
acc_svm_o   = res_svm_o['acc']
acc_svm_pca = res_svm_pca['acc']
acc_svm_ae  = res_svm_ae['acc']

resultados = dict(
    acc_rf_o=acc_rf_o, acc_rf_pca=acc_rf_pca, acc_rf_ae=acc_rf_ae,
    acc_svm_o=acc_svm_o, acc_svm_pca=acc_svm_pca, acc_svm_ae=acc_svm_ae
)
with open('models/acc_results.pkl', 'wb') as f:
    pickle.dump(resultados, f)

resultados_metricas = {
    'RF_original':  res_rf_o,
    'RF_PCA':       res_rf_pca,
    'RF_AE':        res_rf_ae,
    'SVM_original': res_svm_o,
    'SVM_PCA':      res_svm_pca,
    'SVM_AE':       res_svm_ae,
}
with open('models/metrics_results.pkl', 'wb') as f:
    pickle.dump(resultados_metricas, f)
print("[NB05] acc_results.pkl + metrics_results.pkl salvos")

# ─── Testes estatisticos ──────────────────────────────────────────────────────
t_rf_o_pca  = ttest_rel(acc_rf_o, acc_rf_pca)
t_rf_o_ae   = ttest_rel(acc_rf_o, acc_rf_ae)
t_svm_o_pca = ttest_rel(acc_svm_o, acc_svm_pca)
t_svm_o_ae  = ttest_rel(acc_svm_o, acc_svm_ae)

w_rf_o_pca  = wilcoxon(acc_rf_o, acc_rf_pca)
w_rf_o_ae   = wilcoxon(acc_rf_o, acc_rf_ae)
w_svm_o_pca = wilcoxon(acc_svm_o, acc_svm_pca)
w_svm_o_ae  = wilcoxon(acc_svm_o, acc_svm_ae)

friedman = friedmanchisquare(
    acc_rf_o, acc_rf_pca, acc_rf_ae,
    acc_svm_o, acc_svm_pca, acc_svm_ae
)

# Ranking medio Friedman
matriz = np.vstack([acc_rf_o, acc_rf_pca, acc_rf_ae,
                    acc_svm_o, acc_svm_pca, acc_svm_ae]).T
rankings = np.argsort(np.argsort(-matriz, axis=1), axis=1) + 1
ranking_medio = rankings.mean(axis=0)
labels6 = ["RF_original","RF_PCA","RF_AE","SVM_original","SVM_PCA","SVM_AE"]

# ─── Impressao dos resultados ─────────────────────────────────────────────────
sep = "=" * 70
print(f"\n{sep}")
print("TESTES ESTATISTICOS -- random_state=42")
print(sep)
print(f"\nFriedman: chi2 = {friedman.statistic:.4f},  p = {friedman.pvalue:.2e}")

print("\n--- t-pareado (two-sided) ---")
for name, res in [("RF_O   vs RF_PCA ", t_rf_o_pca),
                  ("RF_O   vs RF_AE  ", t_rf_o_ae),
                  ("SVM_O  vs SVM_PCA", t_svm_o_pca),
                  ("SVM_O  vs SVM_AE ", t_svm_o_ae)]:
    sig = "***" if res.pvalue < 0.001 else ("**" if res.pvalue < 0.01 else ("*" if res.pvalue < 0.05 else "ns"))
    print(f"  {name}  t={res.statistic:+7.4f}  p={res.pvalue:.4e}  {sig}")

print("\n--- Wilcoxon (nao-parametrico) ---")
for name, res in [("RF_O   vs RF_PCA ", w_rf_o_pca),
                  ("RF_O   vs RF_AE  ", w_rf_o_ae),
                  ("SVM_O  vs SVM_PCA", w_svm_o_pca),
                  ("SVM_O  vs SVM_AE ", w_svm_o_ae)]:
    sig = "***" if res.pvalue < 0.001 else ("**" if res.pvalue < 0.01 else ("*" if res.pvalue < 0.05 else "ns"))
    print(f"  {name}  W={res.statistic:6.1f}  p={res.pvalue:.4e}  {sig}")

print("\n--- Ranking Medio Friedman ---")
for lbl, rk in zip(labels6, ranking_medio):
    print(f"  {lbl:<16} : {rk:.3f}")

print(f"\n{sep}")
print("TABELA CONSOLIDADA -- 30 repeticoes, random_state=42")
print(sep)
print(f"{'Configuracao':<16} {'Acuracia':>22} {'Precisao':>22} {'Recall':>22} {'F1-score':>22}")
print("-" * 96)
for nome, res in resultados_metricas.items():
    am, as_ = res['acc'].mean(), res['acc'].std()
    pm, ps  = res['prec'].mean(), res['prec'].std()
    rm, rs  = res['rec'].mean(),  res['rec'].std()
    fm, fs  = res['f1'].mean(),   res['f1'].std()
    print(f"{nome:<16}  {am:.4f} +- {as_:.4f}   {pm:.4f} +- {ps:.4f}   {rm:.4f} +- {rs:.4f}   {fm:.4f} +- {fs:.4f}")

print("\n--- Medianas das acuracias ---")
for lbl, res in resultados_metricas.items():
    print(f"  {lbl:<16} : {np.median(res['acc']):.4f}")

print(f"\n--- Loss final autoencoder (epoca 40) ---")
print(f"  {final_loss:.6f}")
