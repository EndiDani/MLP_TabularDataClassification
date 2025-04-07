import pandas as pandas
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Carico il dataset Titanic 
data = sns.load_dataset('titanic')

# Visualizzazione dei dati
print(data.head())

# Gestione dei valori mancanti
imputer = SimpleImputer(strategy='mean') # sostituisce i valori mancanti con la media
data['age'] = imputer.fit_transform(data[['age']])

# Conversione in binario del sesso dei passegeri
data['sex'] = data['sex'].map({'male': 0, 'female': 1})

# Selezione delle feature e l'etichetta
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
X = data[features]
y = data['survived']

# Gestione valori mancanti nelle feature    
X = X.dropna()

# Divisione in training e testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizzazione delle caratteristiche 
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)