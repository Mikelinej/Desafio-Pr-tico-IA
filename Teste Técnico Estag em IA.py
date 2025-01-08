import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = 'Dataset.csv'
df = pd.read_csv(file_path)

num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

num_cols = ['Idade', 'Renda Anual (em $)', 'Tempo no Site (min)']
cat_cols = ['Gênero', 'Anúncio Clicado']

df[num_cols] = num_imputer.fit_transform(df[num_cols])
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

label_encoder = LabelEncoder()
df['Gênero'] = label_encoder.fit_transform(df['Gênero'])
df['Anúncio Clicado'] = label_encoder.fit_transform(df['Anúncio Clicado'])

X = df.drop('Compra (0 ou 1)', axis=1)
y = df['Compra (0 ou 1)']

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

modelo = LogisticRegression(random_state=42)
modelo.fit(X_train_resampled, y_train_resampled)

y_pred = modelo.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {acuracia}\n')
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Compra', 'Compra'], yticklabels=['Não Compra', 'Compra'])
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(df['Idade'], kde=True, color='blue')
plt.title('Distribuição da Idade do Público')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()

df_original = pd.read_csv('Dataset.csv')
plt.figure(figsize=(8, 6))
sns.countplot(x='Anúncio Clicado', hue='Gênero', data=df_original)
plt.title('Proporção de Cliques no Anúncio por Gênero')
plt.xlabel('Clicou no Anúncio')
plt.ylabel('Contagem')
plt.show()
