# Análise de Componentes Principais (PCA)

## 1. Introdução

A **Análise de Componentes Principais (Principal Component Analysis – PCA)** é uma técnica estatística para **redução de dimensionalidade**.  
Ela transforma variáveis possivelmente correlacionadas em **componentes principais (PCs)**, que são novas variáveis **não correlacionadas** e que capturam a maior parte da variância dos dados originais.

---

## 2. Fundamento Matemático

A seguir apresentamos a formulação canônica do PCA para uma matriz de dados $X \in \mathbb{R}^{n \times p}$ (com $n$ observações e $p$ variáveis).  
**Obs.:** As fórmulas estão em blocos isolados `$$...$$` (com linhas em branco antes e depois) para renderizar corretamente no GitHub/VSCode.

### 2.1 Centralização dos dados

$$
X_c = X - \bar{X}
$$

onde $\bar{X}$ é o vetor das médias de cada variável (replicado por linha ao subtrair de $X$).

### 2.2 Matriz de covariância

$$
\Sigma = \frac{1}{n-1}\, X_c^{\mathsf{T}}\, X_c
$$

### 2.3 Decomposição espectral (autovalores e autovetores de $\Sigma$)

$$
\Sigma\, v_i = \lambda_i\, v_i \quad \text{para } i = 1, \dots, p
$$

Os autovetores $v_i$ são **ortogonais** entre si e cada um aponta para uma **direção de máxima variância**.  
Ordenamos $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_p$ e empilhamos os $k$ primeiros autovetores em

$$
V_k = [\,v_1\ v_2\ \dots\ v_k\,] \in \mathbb{R}^{p \times k}.
$$

### 2.4 Projeção no subespaço de dimensão $k$

$$
Z = X_c\, V_k \in \mathbb{R}^{n \times k}
$$

A matriz $Z$ contém as coordenadas dos dados nos **$k$ componentes principais**.

---

## 3. Variância Explicada

Cada componente principal explica uma fração da variância total dada por

$$
\mathrm{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j},
$$

e a variância acumulada até $k$ componentes é

$$
\sum_{i=1}^{k} \mathrm{EVR}_i.
$$

---

## 4. Benefícios no Projeto

| Benefício                             | Impacto                                                |
| ------------------------------------- | ------------------------------------------------------ |
| Redução de dimensionalidade (30 → ~7) | Menor custo, menor ruído e menor risco de overfitting  |
| Componentes não correlacionados       | Modelagem mais estável e interpretável                 |
| Remoção de redundância                | Captura dos padrões estruturais dominantes             |
| Visualização 2D/3D                    | Exploração mais intuitiva de clusters e separabilidade |

---

## 5. Relação com o Pipeline (PCA + Random Forest)

No nosso pipeline, o PCA é aplicado **após a padronização** (via `StandardScaler`) e **antes** do classificador.  
Compararemos o desempenho do **Random Forest** **com** e **sem** PCA para avaliar acurácia, precisão, recall, F1, robustez e tempo de treinamento.

---

## 6. Referências

- JOLLIFFE, I. T. _Principal Component Analysis_. Springer, 2002.
- JAMES, G. et al. _An Introduction to Statistical Learning_. Springer, 2021.
- HAN, J.; KAMBER, M.; PEI, J. _Data Mining: Concepts and Techniques_. Elsevier, 2011.
- PEDREGOSA, F. et al. _Scikit-learn: Machine Learning in Python_. _JMLR_, 12, 2011.
- GERON, A. _Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow_. O’Reilly, 2022.

---

**Autores:** Vanthuir Maia; Luiz Vitor Póvoas  
**Projeto:** Reconhecimento de Padrões — PCA + Random Forest  
**Orientação:** Prof. Fausto Lorenzato — PPGEC/UPE
