import numpy as np

def hybrid_predict(models, X_pca, X_ae):
    rf_pca, svm_pca, rf_ae, svm_ae = models

    pred1 = rf_pca.predict(X_pca)
    pred2 = svm_pca.predict(X_pca)
    pred3 = rf_ae.predict(X_ae)
    pred4 = svm_ae.predict(X_ae)

    preds = np.vstack([pred1, pred2, pred3, pred4]).T

    final_pred = []
    for row in preds:
        values, counts = np.unique(row, return_counts=True)
        winner = values[np.argmax(counts)]
        final_pred.append(winner)

    return np.array(final_pred)
