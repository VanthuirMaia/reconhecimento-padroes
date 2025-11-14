# 🧬 Projeto de Reconhecimento de Padrões — Sistema Híbrido PCA + Autoencoder + Meta-Aprendizagem

Este projeto foi desenvolvido como parte da disciplina **Reconhecimento de Padrões (PPGEC/UPE)**.  
O objetivo é implementar, analisar e comparar **três representações diferentes dos dados** aplicadas a classificadores tradicionais:

- 🔹 **Representação Original**
- 🔹 **PCA (redução linear de dimensionalidade)**
- 🔹 **Autoencoder (redução não linear)**
- 🔹 **Meta-aprendizagem via Grid Search** para otimização dos modelos

Além disso, foram aplicados **testes estatísticos formais** (t-test, Wilcoxon e Friedman) para comprovar a significância dos resultados.

---

## 🎯 Objetivo Geral

Construir um **sistema híbrido completo de reconhecimento de padrões**, integrando:

- Redução de dimensionalidade (PCA e Autoencoder)
- Classificadores (Random Forest e SVM)
- Validação cruzada + Grid Search
- Testes de hipótese
- Visualizações científicas
- Comparação com literatura

Usando o dataset **Breast Cancer Wisconsin (WDBC)**.

---

## 🧩 Estrutura do Projeto

```
reconhecimento-padroes/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_exploracao.ipynb
│   ├── 02_preprocessamento.ipynb
│   ├── 03_modelo_pca_rf.ipynb
│   ├── 04_sistema_hibrido.ipynb       # Autoencoder + RF/SVM + Grid Search
│   ├── 05_testes_estatisticos.ipynb   # t-test, Wilcoxon, Friedman
│   ├── 06_figuras_resultados.ipynb    # Todas as figuras finais
│
├── src/
│   ├── data_processing.py
│   ├── pca_analysis.py
│   ├── model_random_forest.py
│   ├── autoencoder.py                 # Arquitetura e treinamento do AE
│   ├── utils.py
│
├── models/
│   ├── encoder.pkl
│   ├── ae_history.pkl
│   ├── acc_results.pkl
│
├── results/
│   ├── plots/
│   ├── metrics.json
│
├── reports/
│   ├── artigo/
│   ├── resumo_executivo.md
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalação e Execução

### 1️⃣ Criar ambiente virtual

```bash
python -m venv .venv
.venv\Scriptsctivate       # Windows
# ou source .venv/bin/activate  (Linux/Mac)
```

### 2️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

### 3️⃣ Executar notebooks na ordem:

1. **01_exploracao.ipynb**
2. **02_preprocessamento.ipynb**
3. **03_modelo_pca_rf.ipynb**
4. **04_sistema_hibrido.ipynb**
5. **05_testes_estatisticos.ipynb**
6. **06_figuras_resultados.ipynb**

Ou executar pipeline automatizado (se configurado):

```bash
python main.py
```

---

## 📊 Resultados Principais

### Random Forest

| Métrica  | Original | PCA   | AE    |
| -------- | -------- | ----- | ----- |
| Acurácia | 0.947    | 0.921 | 0.921 |

### SVM

| Métrica  | Original | PCA   | AE    |
| -------- | -------- | ----- | ----- |
| Acurácia | 0.982    | 0.956 | 0.938 |

📌 **SVM com dados originais apresentou o melhor desempenho geral.**

---

## 🧪 Testes Estatísticos

Foram aplicados:

- **t-test pareado**
- **Wilcoxon signed-rank**
- **Friedman**

O teste de Friedman resultou em:

```
χ² = 78.38
p < 1e-14
```

➡️ Indica diferença estatisticamente significativa entre os métodos.

---

## 🎨 Figuras Geradas

- Boxplot comparativo das acurácias
- Ranking de Friedman
- Heatmap das diferenças
- PCA 2D
- Autoencoder 3D
- Curva de treinamento do Autoencoder
- Arquitetura visual do Autoencoder

---

## 🧠 Tecnologias Utilizadas

- Python 3.10
- TensorFlow 2.15
- scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
- SciPy (testes estatísticos)
- Jupyter Notebook

---

## 👨‍💻 Autores

**Vanthuir Maia**  
Mestrado em Engenharia da Computação — UPE  
Residência em IA Generativa — UPE  
📧 vnm@ecomp.poli.br  
📧 vanmaiasf@gmail.com

**Luiz Vitor Póvoas**  
Mestrado em Engenharia da Computação — UPE  
📧 lvsp@ecomp.poli.br

---

## 📜 Licença

Este projeto é destinado a **fins acadêmicos e de pesquisa**.  
Uso comercial não autorizado sem permissão dos autores.
