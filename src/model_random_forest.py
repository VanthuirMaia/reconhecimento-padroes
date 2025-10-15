from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold

@dataclass
class RFConfig:
    n_estimators: int = 300 #nº de árvores (mais árvores => mais estável)
    max_depth: Optional[int] = None #None = cresce até a folha pura (cuidado p/ overfit)
    max_features: str | float | int | None = "sqrt" #nº de features avaliadas por split
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    class_weight: Optional[dict | str] = None
    random_state: int = 42
    n_jobs: int = -1 #usa todos os núcleos da máquina

# Constroi o modelo com os hiperparâmetros definidos.
# Idea central do RF: média de várias árvores fracas => modelo robusto.
def build_rf(cfg: RFConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        max_features=cfg.max_features,
        min_samples_split=cfg.min_samples_split,
        min_samples_leaf=cfg.min_samples_leaf,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs
    )

def fit_predict(
    model: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray

) -> np.ndarray:
    # Treina (fit) no conjunto de treino e retorna as predições no teste.
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    # Calcula métricas clássicas de classificação binária.
    # average='binary' supõe y ∈ {0,1} e trata 1 como classe positiva.
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist()
    }

def evaluate_rf_no_pca(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cfg: Optional[RFConfig] = None
) -> Dict[str, Any]:
    #Treina e avalia RF usando todas as features escalonadas (sem PCA).
    cfg = cfg or RFConfig()
    model = build_rf(cfg)
    y_pred = fit_predict(model, X_train_scaled, y_train, X_test_scaled)
    metrics = compute_metrics(y_test, y_pred)
    metrics["settings"] = "no_pca"
    return metrics

def evaluate_rf_with_pca(
    X_train_pca: np.ndarray,
    X_test_pca: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cfg: Optional[RFConfig] = None
) -> Dict[str, Any]:
    #Treina e avalia RF usando features reduzidas via PCA.
    cfg = cfg or RFConfig()
    model = build_rf(cfg)
    y_pred = fit_predict(model, X_train_pca, y_train, X_test_pca)
    metrics = compute_metrics(y_test, y_pred)
    metrics["settings"] = "with_pca"
    metrics["pca_n_components"] = X_train_pca.shape[1]
    return metrics  

def cross_validate_rf(X, y, cfg: RFConfig, use_pca=False, n_components=7, n_splits=5):
    """
    Executa validação cruzada estratificada do Random Forest.
    Retorna médias e desvios das métricas.
    """
    from .pca_analysis import apply_pca
    from .data_processing import preprocess_data
    from sklearn.utils import shuffle

    # Embaralha para evitar padrões de ordenação
    X, y = shuffle(X, y, random_state=cfg.random_state)

    # Define o validador estratificado (mantém proporção de classes)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)

    # Dicionário para armazenar métricas
    results = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Padroniza
        X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

        # Aplica PCA opcional
        if use_pca:
            X_train_scaled, X_test_scaled, _ = apply_pca(
                X_train_scaled, X_test_scaled, n_components=n_components, plot_variance=False
            )

        # Treina e avalia
        model = build_rf(cfg)
        y_pred = fit_predict(model, X_train_scaled, y_train, X_test_scaled)
        metrics = compute_metrics(y_test, y_pred)

        # Armazena resultados
        for key in results.keys():
            results[key].append(metrics[key])

    # Calcula médias e desvios-padrão
    summary = {k: (np.mean(v), np.std(v)) for k, v in results.items()}
    summary["use_pca"] = use_pca
    summary["n_components"] = n_components if use_pca else None
    summary["n_splits"] = n_splits
    summary["raw_results"] = results  # << adiciona isso
    return summary
