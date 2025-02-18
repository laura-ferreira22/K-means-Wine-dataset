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


# checar tipos de dados
wine_df.dtypes

# Distribuição de Genótipos por ordem de frequência
plt.figure(figsize=(15, 12))
sns.countplot(y='Genotypes', data=wine_df, order=wine_df['Genotypes'].value_counts().index)
plt.title('Distribution of Genotypes')
plt.xlabel('Count')
plt.ylabel('Genotypes')
plt.show()


# Mapa de correlação
plt.figure(figsize=(12, 10))
sns.heatmap(dados_numericos.corr(), annot=True, cmap='viridis', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# padronização de dados
ss=StandardScaler()
data_scaled = ss.fit_transform(dados_numericos)


# Função para calcular WCSS

def calcular_wcss(dados_numericos, k_max):
  wcss=[]
  for k in range (1, k_max+1):
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(dados_numericos)
    wcss.append(kmeans.inertia_)
  return wcss

# Definir o número máximo de clusters para testar
k_max = 10

# Calcular WCSS para diferentes valores de k
wcss = calcular_wcss(dados_numericos, k_max)
print(wcss)


# Identificar o número ideal de clusters pelo método Elbow
plt.figure(figsize=(8,6))
plt.plot(range(1, 11), wcss, marker ='o', linestyle = '--')
plt.title("Método Elbow para determinar o número ideal de Clusters")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Soma dos quadrados dentr do cluster (wcss)")
plt.grid(True)
plt.show()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(wine_df.drop(columns=['Genotypes']))

#Obter os rótulos dos clusters e os medoids
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)

# cópia do dataset e adicionando uma coluna cluster
data1 = wine_df.copy()
data1['Cluster'] = kmeans.labels_


# Calculando as distâncias dos pontos ao medoid de seus clusters
distances = pairwise_distances(data_scaled, kmeans.cluster_centers_)
data1['Distance_to_center'] = distances[np.arange(len(distances)), data1['Cluster']]


# Calculando a distância média por cluster (avaliando a coesão)
mean_distances = data1.groupby('Cluster')['Distance_to_center'].mean()
print("Mean distance from points to their kmeans cluster centers:")
print(mean_distances)

# Calculando a distância média geral
overall_mean_distance = data1['Distance_to_center'].mean()
print(f"Overall mean distance: {overall_mean_distance}")


# Redução de dimensionalidade com PCA

pca = PCA(n_components=2)
pca_components = pca.fit_transform(data_scaled)

data_pca = pd.DataFrame(pca_components, columns=['PCA1', 'PCA2'])

data_pca['Cluster'] = data1['Cluster']
data_pca['Genotypes'] = wine_df['Genotypes']
# Visualização dos clusters no espaço PCA
plt.figure(figsize=(12, 10))
plt.scatter(data_pca['PCA1'], data_pca['PCA2'], c=data_pca['Cluster'], cmap='viridis', marker='o', s=50)
# Adicionando os nomes dos genótipos ao gráfico
annotated_genotypes = set()
for i, genotype in enumerate(data_pca['Genotypes']):
    if genotype not in annotated_genotypes:
        plt.text(data_pca['PCA1'].iloc[i], data_pca['PCA2'].iloc[i], genotype,
                 fontsize=9, ha='right', va='bottom', color='black')
        annotated_genotypes.add(genotype)

plt.colorbar(label='Cluster')
plt.title('Visualização de cluster usando PCA com nomes de genótipos')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()


centroids = kmeans.cluster_centers_
centroides_df = pd.DataFrame(centroides, columns=dados_numericos.columns)
print("centroids dos clusters:")
print(centroides_df)

# Diferenças nos atributos entre os centroids
centroides_differences = centroides_df.diff().abs()
print("----------------------------------------------------------------------------")
print("Diferenças entre os centroids:")
print(centroides_differences)
print("----------------------------------------------------------------------------")
print("Maiores diferenças entre os centroids:")
print(centroides_differences.max())

silhouette_coef = silhouette_score(data_scaled, labels)
print(f'Coeficiente da silhueta : {silhouette_coef}')

print(f'inércia : {kmeans.inertia_}')