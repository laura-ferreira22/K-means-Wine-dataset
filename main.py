import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, jaccard_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

# Carregar o dataset
from google.colab import drive
drive.mount('/content/drive')

wine_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Analysis.csv')

atributos = wine_df.columns
print(atributos)

# Resumo estístico básico
wine_df.describe()

# removendo a coluna 'Unnamed: 0'
wine_df.drop(columns=['Unnamed: 0'], inplace=True)
wine_df.head(5)

# checar os valores nulos
wine_df.isnull().sum()


# medidas de tendência central
dados_numericos = wine_df.select_dtypes(include=[float,int])


for coluna in dados_numericos.columns:
    print(f"Média da coluna {coluna}: {dados_numericos[coluna].mean()}")
    print(f"Mediana da coluna {coluna}: {dados_numericos[coluna].median()}")
    print(f"Moda da coluna {coluna}: {dados_numericos[coluna].mode()[0]}")
    print()


# medidas de tendência central
dados_numericos = wine_df.select_dtypes(include=[float,int])


for coluna in dados_numericos.columns:
    print(f"Média da coluna {coluna}: {dados_numericos[coluna].mean()}")
    print(f"Mediana da coluna {coluna}: {dados_numericos[coluna].median()}")
    print(f"Moda da coluna {coluna}: {dados_numericos[coluna].mode()[0]}")
    print()


# medidas de tendência central
dados_numericos = wine_df.select_dtypes(include=[float,int])


for coluna in dados_numericos.columns:
    print(f"Média da coluna {coluna}: {dados_numericos[coluna].mean()}")
    print(f"Mediana da coluna {coluna}: {dados_numericos[coluna].median()}")
    print(f"Moda da coluna {coluna}: {dados_numericos[coluna].mode()[0]}")
    print()


# Teste de Normalidade (Shapiro-Wilk)
for coluna in dados_numericos.columns:
    stat, p = stats.shapiro(dados_numericos[coluna])
    print(f"\nTeste de Normalidade (Shapiro-Wilk) {coluna}: ")
    print(f"Estatística de teste {coluna}: {stat}, p-valor {p}")


    # Interpretação do p-valor
    # alfa grau de significância

    alfa = 0.05

    if p > alfa:
      print('a distribuição é normal')
    else:
      print('a distribuição não é normal')



# Teste de Normalidade (Shapiro-Wilk)
for coluna in dados_numericos.columns:
    stat, p = stats.shapiro(dados_numericos[coluna])
    print(f"\nTeste de Normalidade (Shapiro-Wilk) {coluna}: ")
    print(f"Estatística de teste {coluna}: {stat}, p-valor {p}")


    # Interpretação do p-valor
    # alfa grau de significância

    alfa = 0.05

    if p > alfa:
      print('a distribuição é normal')
    else:
      print('a distribuição não é normal')



# Gráfico de frequência
for coluna in dados_numericos.columns:
  plt.figure(figsize=(10,5))
  sns.histplot(dados_numericos[coluna],bins = 30, kde = False)
  plt.title(f'Gráfico de frequência da {coluna}')
  plt.show()


# Boxplot
plt.figure(figsize=(16,8))
sns.boxplot(data=dados_numericos)
plt.title("Boxplot Wine Genotype Classification antes do tratamento")
plt.show()


# Boxplot
plt.figure(figsize=(16,8))
sns.boxplot(data=dados_numericos)
plt.title("Boxplot Wine Genotype Classification antes do tratamento")
plt.show()