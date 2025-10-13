# Padronização de Dados (StandardScaler)

## 1. Introdução

A padronização é uma etapa fundamental no pré-processamento de dados para modelos de aprendizado de máquina, especialmente em métodos baseados em **distância** ou **variância**, como o **PCA (Principal Component Analysis)**.  
Seu objetivo é garantir que todas as variáveis numéricas possuam **escala comparável**, evitando que atributos com magnitudes elevadas dominem a análise.

---

## 2. Formulação Matemática

A transformação aplicada pelo `StandardScaler` segue a fórmula:

$$
z_i = \frac{x_i - \mu}{\sigma}
$$

onde:

- \( x_i \) é o valor original da variável;
- \( \mu \) é a média da variável;
- \( \sigma \) é o desvio padrão da variável;
- \( z_i \) é o valor padronizado.

O resultado é uma nova variável com **média zero** e **desvio padrão unitário**.

---

## 3. O que é alterado e o que é preservado

| Aspecto               | Comportamento após padronização                                  |
| --------------------- | ---------------------------------------------------------------- |
| Escala                | Alterada — todas as variáveis passam a ter média 0 e desvio 1    |
| Forma da distribuição | Preservada (a distribuição mantém sua forma e ordem dos valores) |
| Informação relativa   | Preservada — apenas a unidade de medida muda                     |
| Unidade original      | Removida (as variáveis passam a ser medidas em desvios padrão)   |

---

## 4. Motivação para o PCA

O PCA busca **direções de máxima variância** no espaço dos dados.  
Quando as variáveis possuem escalas muito diferentes (ex.: área ≫ textura), as de maior magnitude dominam o cálculo da variância total.

A padronização elimina esse viés, permitindo que o PCA identifique **padrões reais de correlação** entre as variáveis, e não apenas diferenças de unidade.

---

## 5. Interpretação Geométrica

Antes da padronização, cada eixo do espaço de atributos pode ter unidades diferentes (ex.: milímetros, centímetros, kg).  
Após a transformação, todos os eixos passam a ter **média zero e amplitude semelhante**, o que equivale a **rotacionar e centralizar o espaço** em torno da origem.

Isso garante que o PCA — ao decompor a matriz de covariância — produza componentes principais que refletem **relações estruturais** entre as variáveis, e não apenas escalas arbitrárias.

---

## 6. Exemplo Numérico

| Valor original | Média | Desvio | Valor padronizado |
| -------------- | ----- | ------ | ----------------- |
| 10             | 17.5  | 6.45   | -1.16             |
| 15             | 17.5  | 6.45   | -0.39             |
| 20             | 17.5  | 6.45   | +0.39             |
| 25             | 17.5  | 6.45   | +1.16             |

A ordem e a dispersão relativa são mantidas, mas a escala é uniformizada.

---

## 7. Relação com o Projeto

No projeto **Reconhecimento de Padrões — PCA + Random Forest**, a padronização foi aplicada via `StandardScaler` antes da decomposição em componentes principais.  
Essa etapa assegura que todas as 30 variáveis do dataset **Breast Cancer Wisconsin** contribuam igualmente para o cálculo da variância total.

---

## 8. Referências Bibliográficas

- JAMES, G. et al. _An Introduction to Statistical Learning: with Applications in R_. 2. ed. Springer, 2021.
- HAN, J.; KAMBER, M.; PEI, J. _Data Mining: Concepts and Techniques_. 3. ed. Elsevier, 2011.
- PEDREGOSA, F. et al. _Scikit-learn: Machine Learning in Python_. _Journal of Machine Learning Research_, v. 12, p. 2825–2830, 2011.
- GERON, A. _Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow_. 3. ed. O’Reilly, 2022.

---

**Autores:** Vanthuir Maia; Luiz Vitor Póvoas  
**Projeto:** Reconhecimento de Padrões — PCA + Random Forest  
**Orientação:** Prof. Fausto Lorenzato — PPGEC/UPE
