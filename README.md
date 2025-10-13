# Projeto de Reconhecimento de Padrões — PCA + Random Forest

Este projeto foi desenvolvido como parte da disciplina **Reconhecimento de Padrões (PPGEC/UPE)**, com o objetivo de comparar o desempenho de um classificador **Random Forest** antes e depois da aplicação de **PCA (Principal Component Analysis)**, uma técnica de redução de dimensionalidade.

---

## 🎯 Objetivo

Desenvolver um sistema de **classificação supervisionada** utilizando o dataset **Breast Cancer Wisconsin**, avaliando o impacto da redução de dimensionalidade via **PCA** sobre o desempenho do modelo **Random Forest**.

---

## 🧩 Estrutura do Projeto

```
reconhecimento-padroes/
│
├── data/
│   ├── raw/                # Dados originais
│   ├── processed/          # Dados tratados
│
├── notebooks/
│   ├── 01_exploracao.ipynb
│   ├── 02_preprocessamento.ipynb
│   ├── 03_modelo_pca_rf.ipynb
│
├── src/
│   ├── data_processing.py   # Funções de limpeza e normalização
│   ├── pca_analysis.py      # Funções de PCA
│   ├── model_rf.py          # Treino e avaliação Random Forest
│
├── results/
│   ├── metrics.json         # Resultados numéricos
│   ├── plots/               # Gráficos salvos (matriz confusão, PCA 2D etc.)
│
├── reports/
│   ├── artigo/
│   │   └── artigo.tex       # Versão em LaTeX do artigo
│   └── resumo_executivo.md
│
├── .gitignore
├── requirements.txt
└── main.py
```

---

## ⚙️ Instalação e Execução

### 1️⃣ Criar ambiente virtual e instalar dependências

```bash
python -m venv .venv
.venv\Scripts\activate    # (Windows)
# ou source .venv/bin/activate (Linux/Mac)

pip install -r requirements.txt
```

### 2️⃣ Executar o projeto principal

```bash
python main.py
```

---

## 📊 Métricas esperadas

| Métrica  | Sem PCA | Com PCA |
| -------- | ------- | ------- |
| Acurácia |         |         |
| Precisão |         |         |
| Recall   |         |         |
| F1-Score |         |         |

---

## 🧠 Tecnologias Utilizadas

- Python 3.11+
- scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 👨‍💻 Autores

**Vanthuir Maia**  
Mestrado em Engenharia da Computação — UPE  
Residência em IA Generativa — UPE  
📧 [Contato profissional](mailto:vnm@ecomp.poli.br)
📧 [Contato profissional](mailto:vanmaiasf@gmail.com)

**Luiz Vitor**
Mestrado em Engenharia da Computação — UPE  
📧 [Contato profissional](mailto:lvsp@ecomp.poli.br)

---

## 📜 Licença

Este projeto é destinado a fins **acadêmicos e de pesquisa**.  
O uso comercial não é autorizado sem o consentimento do autor.
