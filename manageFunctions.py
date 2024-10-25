def create_binary_datasets(df, target_column):
  """
  Cria 4 datasets binários a partir de um dataset com uma coluna de target multiclasse.

  Args:
    df: O DataFrame pandas contendo os dados.
    target_column: O nome da coluna de target.

  Returns:
    Uma lista de 4 DataFrames, cada um representando um dataset binário para uma classe.
  """
  datasets = {}
  unique_classes = df[target_column].unique()
  for target_class in unique_classes:
        df_copy = df.copy()  
        df_copy['target'] = (df_copy[target_column] == target_class).astype(int)  
        df_copy = df_copy.drop(target_column, axis=1)
        datasets['dataset_'+ str(target_class)] = df_copy

  return datasets

datasets = create_binary_datasets(df, 'estadofinal')#ex de uso


from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler

def apply_transforms_and_split(datasets, target):
    """
    Aplica as transformações necessárias aos datasets.

    Args:
        datasets: Uma lista de DataFrames contendo os dados.
        target: O nome da coluna de target.
        
    Returns:
        Um dicionário com os conjuntos de dados transformados.
    """
    df_splits_and_transformed = {}
    for i, dataset in enumerate(datasets):
        CATEGORICAL_FEATURES = [col for col in dataset.columns if dataset.dtypes[col] == 'object']
        NUMERIC_FEATURE = [col for col in dataset.columns if col not in CATEGORICAL_FEATURES + [target]]
        
        X = dataset.drop(target, axis=1)
        y = dataset[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        encoder = TargetEncoder()
        X_train[CATEGORICAL_FEATURES] = encoder.fit_transform(X_train[CATEGORICAL_FEATURES], y_train)
        X_test[CATEGORICAL_FEATURES] = encoder.transform(X_test[CATEGORICAL_FEATURES])
        
        sc = StandardScaler()
        sc.fit(X_train[CATEGORICAL_FEATURES + NUMERIC_FEATURE])
        X_train[CATEGORICAL_FEATURES + NUMERIC_FEATURE] = sc.transform(X_train[CATEGORICAL_FEATURES + NUMERIC_FEATURE])
        X_test[CATEGORICAL_FEATURES + NUMERIC_FEATURE] = sc.transform(X_test[CATEGORICAL_FEATURES + NUMERIC_FEATURE])
        
        df_splits_and_transformed[f'dataset_{i}'] = (X_train, X_test, y_train, y_test)
    
    return df_splits_and_transformed

    datasets_transformed = apply_transforms_and_split(datasets.values(), 'target')#ex de uso