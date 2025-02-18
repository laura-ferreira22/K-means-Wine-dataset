# Análise e Clusterização de Genótipos de Vinho

Este projeto realiza uma análise exploratória e clusterização de genótipos de vinho com base em diversas variáveis numéricas. Utiliza bibliotecas de machine learning para normalização dos dados, testes estatísticos e agrupamento por meio de K-Means e PCA.

## Estrutura do Código

### 1. Importação de Bibliotecas
O código utiliza bibliotecas como Pandas, NumPy, Seaborn, Matplotlib, Scipy e Scikit-learn para processamento, visualização e clusterização dos dados.

### 2. Carregamento dos Dados
Os dados são carregados a partir do Google Drive e armazenados em um DataFrame Pandas.

### 3. Limpeza e Tratamento dos Dados
- Remoção de colunas desnecessárias
- Verificação de valores nulos
- Análise estatística básica (média, mediana e moda)
- Teste de normalidade (Shapiro-Wilk)

### 4. Visualização de Dados
- Histogramas para distribuição das variáveis
- Boxplots para identificar outliers
- Mapa de correlação entre variáveis
- Distribuição dos genótipos

### 5. Clusterização com K-Means
- Padronização dos dados usando `StandardScaler`
- Determinação do número ideal de clusters pelo método Elbow
- Aplicação do algoritmo K-Means e avaliação dos clusters

### 6. Redução de Dimensionalidade com PCA
- Aplica PCA para reduzir os dados a duas dimensões
- Visualização dos clusters com scatter plot

### 7. Avaliação dos Clusters
- Distâncias médias dos pontos aos centróides
- Diferenças entre os atributos dos clusters
- Cálculo do coeficiente da silhueta

## Como Executar o Projeto
1. Subir os arquivos no Google Drive
2. Conectar o Google Drive ao Google Colab
3. Executar todas as células do notebook

## Requisitos
Instalar as bibliotecas necessárias, caso ainda não estejam disponíveis:
```md
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

## Resultados Esperados
- Identificação de clusters distintos de genótipos de vinho
- Visualização das diferenças entre grupos
- Métrica de avaliação de coesão dos clusters

## Autor
Este código foi desenvolvido para fins de análise de dados e aprendizado de métodos de clusterização e redução de dimensionalidade.

