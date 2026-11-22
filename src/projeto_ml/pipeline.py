import pandas as pd
import numpy as np

from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    SmartCorrelatedSelection
)
from feature_engine.pipeline import Pipeline  # <--- CORREÇÃO: Pipeline do feature-engine
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2
RANDOM_STATE = SEED


class Pipeline_Data:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X          # <--- CORREÇÃO: sem vírgula
        self.y = y          # <--- CORREÇÃO: sem vírgula

    def split_data(self):
        # 1. Split inicial (Treino 60%, Temp 40%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y, test_size=0.5, random_state=SEED, stratify=self.y
        )
        
        # 2. Split do Temp (Validação 20%, Teste 20%)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
        )

        # 3. Pipeline de limpeza
        pipeline = Pipeline([
            ('Constantes', DropConstantFeatures(tol=0.95, missing_values='ignore')),
            ('Correlacao', SmartCorrelatedSelection(
                threshold=0.85, 
                method='spearman', 
                selection_method='variance'
            )),
            ('Duplicadas', DropDuplicateFeatures())
        ])

        # 4. Fit no treino, transform nos demais
        X_train_t = pipeline.fit_transform(X_train)
        X_val_t = pipeline.transform(X_val)
        X_test_t = pipeline.transform(X_test)

        # 5. Log das features removidas
        print(f"Shape original do treino: {X_train.shape}")
        print(f"Shape após limpeza: {X_train_t.shape}\n")

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

        # 6. Retorna os dados transformados e os originais (para debug)
        return (
            X_train_t, y_train,
            X_val_t, y_val,
            X_test_t, y_test,
            pipeline  # <--- BÔNUS: retornar o pipeline treinado
        )