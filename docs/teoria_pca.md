# Análise de Componentes Principais (PCA)

## 1. Introdução

A **Análise de Componentes Principais (Principal Component Analysis – PCA)** é uma técnica estatística amplamente utilizada para **redução de dimensionalidade**.  
Seu objetivo é transformar um conjunto de variáveis possivelmente correlacionadas em um **novo conjunto de variáveis não correlacionadas**, chamadas **componentes principais (PCs)**, que capturam a maior parte da variância dos dados originais.

Essa transformação permite representar os dados em um espaço de menor dimensão **sem perda significativa de informação**, reduzindo ruído e redundância — fatores essenciais em tarefas de **reconhecimento de padrões** e **aprendizado supervisionado**.

---

## 2. Fundamento Matemático

1. **Centralização dos dados:**

$$
X_c = X - \bar{X}
$$

onde \( \bar{X} \) é o vetor das médias de cada variável.

2. **Cálculo da matriz de covariância:**

$$
\Sigma = \frac{1}{n-1} X_c^T X_c
$$

3. **Decomposição espectral:**

Calculam-se os **autovalores** (\( \lambda_i \)) e **autovetores** (\( v_i \)) de \( \Sigma \).  
Cada autovetor representa uma **direção de máxima variância** no espaço dos dados.

4. **Ordenação e projeção:**

$$
Z = X_c V_k
$$

onde \( V_k \) contém os \( k \) autovetores associados aos maiores autovalores.

---

## 3. Interpretação Geométrica

Geometricamente, o PCA pode ser visto como uma **rotação dos eixos originais** de forma a alinhar o novo eixo principal com a **direção de maior variância** dos dados.

- O **1º componente principal** aponta na direção de maior dispersão.
- O **2º componente**, ortogonal ao primeiro, captura a maior variação remanescente.
- E assim por diante, até completar o número de variáveis originais.

Essa rotação não altera a relação entre os pontos, apenas muda o sistema de coordenadas, tornando possível **descrever o mesmo conjunto de dados com menos dimensões relevantes**.

---

## 4. Variância Explicada e Curva do Cotovelo

Cada componente principal explica uma fração da variância total dos dados.  
A soma cumulativa dessas frações é chamada de **variância acumulada explicada**.

A **curva de cotovelo (Elbow Plot)** mostra a relação entre o número de componentes e a variância acumulada.  
O ponto de inflexão da curva indica a quantidade ideal de componentes a manter — onde a adição de novos componentes traz **ganhos marginais mínimos**.

No caso do **dataset Breast Cancer Wisconsin**, os resultados indicam:

| Nº de Componentes | Variância Explicada Individual | Variância Acumulada |
| ----------------- | ------------------------------ | ------------------- |
| 1                 | 44,4%                          | 44,4%               |
| 2                 | 18,9%                          | 63,3%               |
| 3                 | 9,5%                           | 72,8%               |
| 4                 | 6,7%                           | 79,5%               |
| 5                 | 5,5%                           | 85,0%               |
| 7                 | ≈ 95%                          | ≈ 95%               |

A partir de **7 componentes**, já é possível explicar cerca de **95% da variância total**, justificando a redução de 30 para 7 dimensões.

---

## 5. Benefícios da Aplicação do PCA

| Benefício                                | Impacto no Projeto                                       |
| ---------------------------------------- | -------------------------------------------------------- |
| Redução da dimensionalidade              | Simplifica o modelo e reduz o custo computacional        |
| Remoção de correlação entre variáveis    | Melhora a estabilidade e interpretabilidade do modelo    |
| Eliminação de ruído                      | Reduz variância indesejada e melhora a generalização     |
| Visualização                             | Permite representação 2D/3D de dados complexos           |
| Eficiência no aprendizado supervisionado | Melhora desempenho de classificadores como Random Forest |

---

## 6. Relação com o Projeto

No projeto **Reconhecimento de Padrões — PCA + Random Forest**, o PCA foi aplicado após a padronização dos dados para identificar as **direções principais de variação** no conjunto _Breast Cancer Wisconsin_.  
Com base no gráfico de variância explicada acumulada, foram selecionados **7 componentes principais**, que retêm aproximadamente **95% da informação total**.

Esses componentes foram então utilizados como entrada no modelo **Random Forest**, permitindo comparar o desempenho entre os modelos **com e sem redução de dimensionalidade**.

---

## 7. Referências Bibliográficas

- JOLLIFFE, I. T. _Principal Component Analysis_. Springer, 2002.
- JAMES, G. et al. _An Introduction to Statistical Learning: with Applications in R_. Springer, 2021.
- HAN, J.; KAMBER, M.; PEI, J. _Data Mining: Concepts and Techniques_. Elsevier, 2011.
- PEDREGOSA, F. et al. _Scikit-learn: Machine Learning in Python_. _Journal of Machine Learning Research_, v. 12, p. 2825–2830, 2011.
- GERON, A. _Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow_. O’Reilly, 2022.

---

**Autores:** Vanthuir Maia; Luiz Vitor Póvoas  
**Projeto:** Reconhecimento de Padrões — PCA + Random Forest  
**Orientação:** Prof. Fausto Lorenzato — PPGEC/UPE
