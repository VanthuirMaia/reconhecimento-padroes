# Random Forest — Fundamento e Aplicação

## 1. Introdução

O **Random Forest (Floresta Aleatória)** é um dos algoritmos de aprendizado supervisionado mais robustos e versáteis da atualidade.  
Seu princípio básico consiste em combinar várias **árvores de decisão independentes** para formar um modelo coletivo (_ensemble_) mais estável e preciso.

A ideia é simples: uma única árvore tende a superajustar os dados (alto _overfitting_), enquanto uma floresta de árvores, construída a partir de amostras e variáveis aleatórias, reduz essa variância e melhora a generalização.

---

## 2. Fundamento Teórico

O Random Forest é uma técnica de **bagging** (_Bootstrap Aggregating_), proposta por _Leo Breiman (2001)_.  
O método combina a ideia de **amostragem aleatória** (bootstrapping) e **subespaços de atributos** (random feature selection).

Cada árvore de decisão é treinada com um subconjunto aleatório de amostras e de variáveis, gerando modelos ligeiramente diferentes entre si.  
Durante a inferência, cada árvore fornece uma predição, e o resultado final é obtido por **votação majoritária** (classificação) ou **média** (regressão).

---

## 3. Formulação Matemática

### 3.1 Treinamento

Para um conjunto de dados $D = \\{(x_i, y_i)\\}_{i=1}^n$ com $n$ amostras e $p$ variáveis:

1. Sorteiam-se $B$ subconjuntos $D_b$ de $D$, com reposição (_bootstrap_);
2. Para cada subconjunto $D_b$, treina-se uma árvore $T_b(x)$ escolhendo aleatoriamente $m < p$ variáveis a cada divisão;
3. Cada árvore cresce até atingir um critério de parada (profundidade máxima, nº mínimo de amostras, etc).

### 3.2 Predição

Para uma nova amostra $x$, cada árvore $T_b(x)$ gera uma predição.  
O resultado final é obtido por:

$$
\hat{y} = \text{mode}\\{T_1(x), T_2(x), \dots, T_B(x)\\}
$$

onde $B$ é o número total de árvores.

A diversidade entre as árvores é o que confere ao Random Forest sua capacidade de **reduzir a variância sem aumentar o viés**.

---

## 4. Hiperparâmetros Principais

| Parâmetro                                | Descrição                           | Efeito prático                                             |
| ---------------------------------------- | ----------------------------------- | ---------------------------------------------------------- |
| `n_estimators`                           | Nº de árvores                       | Mais árvores → maior estabilidade e menor variância        |
| `max_depth`                              | Profundidade máxima                 | Limita o crescimento de cada árvore (controla overfitting) |
| `max_features`                           | Nº de variáveis avaliadas por split | Aumenta aleatoriedade, reduz correlação entre árvores      |
| `min_samples_split` / `min_samples_leaf` | Critérios mínimos de divisão        | Influenciam tamanho e complexidade das árvores             |
| `class_weight`                           | Peso das classes                    | Corrige desbalanceamentos                                  |
| `random_state`                           | Semente aleatória                   | Garante reprodutibilidade                                  |

---

## 5. Interpretação e Importância das Variáveis

Uma vantagem do Random Forest é a **estimativa de importância das variáveis**, obtida pela redução média da impureza (por exemplo, Gini ou Entropia) causada por cada atributo ao longo da floresta.

Essa medida permite identificar **quais variáveis mais contribuem para as decisões do modelo**, fornecendo insights interpretáveis, mesmo em conjuntos de dados de alta dimensionalidade.

---

## 6. Aplicação no Projeto PCA + RF

Neste projeto, utilizamos o Random Forest para comparar dois cenários:

| Cenário           | Descrição                                      | Nº de Features | Desempenho (F1) |
| ----------------- | ---------------------------------------------- | -------------- | --------------- |
| **Sem PCA**       | Modelo com 30 variáveis originais padronizadas | 30             | ≈ 0.958         |
| **Com PCA (k=7)** | Modelo com 7 componentes principais            | 7              | ≈ 0.937         |

A diferença de desempenho é pequena (~2%), mostrando que a redução de dimensionalidade via PCA simplificou o modelo sem comprometer significativamente a performance.

### Interpretação:

- O modelo **sem PCA** retém mais variação, mas é mais complexo.
- O modelo **com PCA** perde pequena precisão, mas ganha estabilidade e interpretabilidade.
- Ambos demonstram **alta capacidade de generalização** no conjunto _Breast Cancer Wisconsin_.

---

## 7. Referências

- BREIMAN, L. _Random Forests_. _Machine Learning_, 45(1), 5–32, 2001.
- GERON, A. _Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow_. O’Reilly, 2022.
- JAMES, G. et al. _An Introduction to Statistical Learning_. Springer, 2021.
- PEDREGOSA, F. et al. _Scikit-learn: Machine Learning in Python_. _JMLR_, 12, 2011.

---

**Autores:** Vanthuir Maia; Luiz Vitor Póvoas  
**Projeto:** Reconhecimento de Padrões — PCA + Random Forest  
**Orientação:** Prof. Fausto Lorenzato — PPGEC/UPE
