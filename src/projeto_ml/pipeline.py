import pandas as pd
import numpy as np

from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    SmartCorrelatedSelection,
    
)

from feature_engine.imputation import CategoricalImputer

from feature_engine.pipeline import Pipeline 
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2
RANDOM_STATE = SEED


class Pipeline_Data:
    def __init__(self, X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE):
        self.X = X          
        self.y = y 
        self.test_size = test_size

    def split_data(self):
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=SEED, stratify=self.y
        )
        
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
        )

        
        pipeline = Pipeline([
            ('imputer_cat', CategoricalImputer(imputation_method='missing', fill_value='SEM_INFORMACAO')),
            ('Constantes', DropConstantFeatures(tol=0.95, missing_values='ignore')),
            ('Correlacao', SmartCorrelatedSelection(
                threshold=0.85, 
                method='spearman', 
                selection_method='variance'
            )),
            ('Duplicadas', DropDuplicateFeatures())
        ])

        
        X_train_t = pipeline.fit_transform(X_train)
        X_val_t = pipeline.transform(X_val)
        X_test_t = pipeline.transform(X_test)

        
        print(f"Shape original do treino: {X_train.shape}")
        print(f"Shape após limpeza: {X_train_t.shape}\n")

        print(f"Shape test: {X_test.shape}\n")
        

        print(f"Shape val: {X_val.shape}\n")
        

        constantes = pipeline.named_steps['Constantes']
        print("--- Features Constantes Removidas ---")
        print(constantes.features_to_drop_)

        correlacao = pipeline.named_steps['Correlacao']
        print("\n--- Features Correlacionadas Removidas ---")
        print(correlacao.features_to_drop_)

        print(f"\n--- Pares de Features Correlacionadas (Correlação > 0.85) ---")
        print(correlacao.correlated_feature_sets_)

        duplicadas = pipeline.named_steps['Duplicadas']
        print("\n--- Features Duplicadas Removidas ---")
        print(duplicadas.features_to_drop_)

        print("\n--- Grupos de Features Duplicadas ---")
        print(duplicadas.duplicated_feature_sets_)

        
        return (
            X_train_t, y_train,
            X_val_t, y_val,
            X_test_t, y_test,
            pipeline 
        )