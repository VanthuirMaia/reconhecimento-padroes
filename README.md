# 🧬 Projeto de Reconhecimento de Padrões — PCA + Random Forest

Este projeto foi desenvolvido como parte da disciplina **Reconhecimento de Padrões (PPGEC/UPE)**.  
O objetivo é comparar o desempenho de um classificador **Random Forest** com e sem a aplicação de **PCA (Principal Component Analysis)**, uma técnica clássica de **redução de dimensionalidade**.

---

## 🎯 Objetivo

Construir um sistema de **classificação supervisionada** usando o dataset **Breast Cancer Wisconsin**, avaliando:

- A influência do **PCA** na performance do **Random Forest**;
- O equilíbrio entre **acurácia** e **simplicidade do modelo**;
- O impacto da **redução de dimensionalidade (30 → 7 componentes)** sobre as métricas de avaliação.

---

## 🧩 Estrutura do Projeto

```
reconhecimento-padroes/
│
├── data/
│   ├── raw/                # Dados originais (brutos)
│   ├── processed/          # Dados padronizados e prontos para modelagem
│
├── notebooks/
│   ├── 01_exploracao.ipynb          # Análise exploratória inicial
│   ├── 02_preprocessamento.ipynb    # Padronização e PCA
│   ├── 03_modelo_pca_rf.ipynb       # Treinamento e comparação dos modelos
│
├── src/
│   ├── data_processing.py           # Funções de carregamento e pré-processamento
│   ├── pca_analysis.py              # Aplicação e visualização do PCA
│   ├── model_random_forest.py       # Modelagem, avaliação e validação cruzada
│
├── results/
│   ├── metrics.json                 # Métricas quantitativas
│   ├── plots/                       # Gráficos (Matriz de Confusão, Boxplots, etc.)
│
├── reports/
│   ├── docs_teoricos/               # Explicações matemáticas (PCA e Pré-processamento)
│   ├── artigo/                      # Versão em LaTeX para submissão
│   └── resumo_executivo.md          # Resumo técnico do projeto
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
.venv\Scripts\activate    # Windows
# ou source .venv/bin/activate  (Linux/Mac)

pip install -r requirements.txt
```

### 2️⃣ Executar notebooks de forma sequencial

1. **01_exploracao.ipynb** → visualização e entendimento do dataset
2. **02_preprocessamento.ipynb** → normalização e PCA
3. **03_modelo_pca_rf.ipynb** → comparação entre os modelos

ou, se preferir rodar o pipeline completo via script:

```bash
python main.py
```

---

## 📊 Principais Resultados

| Métrica      | Sem PCA | Com PCA (k=7) |
| ------------ | ------- | ------------- |
| **Acurácia** | 0.947   | 0.921         |
| **Precisão** | 0.958   | 0.944         |
| **Recall**   | 0.958   | 0.931         |
| **F1-Score** | 0.958   | 0.937         |

> 🔍 O PCA reduziu a dimensionalidade de 30 para 7 atributos,  
> mantendo desempenho semelhante — o que demonstra sua eficiência  
> em cenários de alta dimensionalidade e baixo custo computacional.

---

## 🧠 Tecnologias Utilizadas

- **Python 3.12**
- **scikit-learn** — modelagem e métricas
- **Pandas / NumPy** — manipulação de dados
- **Matplotlib / Seaborn** — visualização científica
- **Jupyter Notebook** — experimentação e reprodutibilidade

---

## 👨‍💻 Autores

**Vanthuir Maia**  
Mestrado em Engenharia da Computação — UPE  
Residência em IA Generativa — UPE  
📧 [vnm@ecomp.poli.br](mailto:vnm@ecomp.poli.br)  
📧 [vanmaiasf@gmail.com](mailto:vanmaiasf@gmail.com)

**Luiz Vitor Póvoas**  
Mestrado em Engenharia da Computação — UPE  
📧 [lvsp@ecomp.poli.br](mailto:lvsp@ecomp.poli.br)

---

## 📜 Licença

Este projeto é destinado a **fins acadêmicos e de pesquisa**.  
Uso comercial não autorizado sem o consentimento dos autores.
