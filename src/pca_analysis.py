from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Aplica PCA aos dados e retorna as versões transformadas.Também pode gerar um gráfico da variância explicada.

def apply_pca(X_train, X_test, n_components=None, plot_variance=True):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    if plot_variance:
        explained_var = np.cumsum(pca.explained_variance_ratio_)
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(explained_var) + 1), explained_var, marker='o')
        plt.xlabel('Número de Componentes Principais')
        plt.ylabel('Variância Acumulada Explicada')
        plt.title('PCA - Variância Explicada Acumulada')
        plt.grid(True)
        plt.show()
    
    return X_train_pca, X_test_pca, pca