import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from sklearn.metrics import (
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve, 
    auc, 
    f1_score, 
    precision_score, 
    recall_score, 
    accuracy_score, 
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay
)

class AvaliacaoModelo:
    def __init__(self, modelo, X_train, y_train, X_holdout, y_holdout, nome_modelo: str = "Modelo"):
        self.modelo = modelo
        self.X_train = X_train
        self.y_train = y_train
        self.X_holdout = X_holdout
        self.y_holdout = y_holdout
        self.nome_modelo = nome_modelo
        
        # Fazendo as predições de Treino
        self.y_pred_train = modelo.predict(X_train)
        self.y_prob_train = modelo.predict_proba(X_train)[:, 1]
        
        # Fazendo as predições de Holdout (Teste Final)
        self.y_pred_holdout = modelo.predict(X_holdout)
        self.y_prob_holdout = modelo.predict_proba(X_holdout)[:, 1]

    def calcular_metricas(self) -> pd.DataFrame:
        """
        Calcula as métricas de Treino e Holdout e gera uma tabela comparativa (Gap).
        """
        metricas = {
            'Métrica': ['ROC-AUC', 'PR-AUC', 'Logloss', 'F1-Score (Churn)', 'Precision (Churn)', 'Recall (Churn)', 'Acurácia'],
            'Treino': [
                roc_auc_score(self.y_train, self.y_prob_train),
                auc(*precision_recall_curve(self.y_train, self.y_prob_train)[1::-1]),
                log_loss(self.y_train, self.y_prob_train),
                f1_score(self.y_train, self.y_pred_train),
                precision_score(self.y_train, self.y_pred_train),
                recall_score(self.y_train, self.y_pred_train),
                accuracy_score(self.y_train, self.y_pred_train)
            ],
            'Holdout (Teste)': [
                roc_auc_score(self.y_holdout, self.y_prob_holdout),
                auc(*precision_recall_curve(self.y_holdout, self.y_prob_holdout)[1::-1]),
                log_loss(self.y_holdout, self.y_prob_holdout),
                f1_score(self.y_holdout, self.y_pred_holdout),
                precision_score(self.y_holdout, self.y_pred_holdout),
                recall_score(self.y_holdout, self.y_pred_holdout),
                accuracy_score(self.y_holdout, self.y_pred_holdout)
            ]
        }
        
        df_metricas = pd.DataFrame(metricas)
        # Adicionando a coluna de Gap (diferença)
        df_metricas['Gap (Treino - Holdout)'] = df_metricas['Treino'] - df_metricas['Holdout (Teste)']
        
        print(f"{'='*60}")
        print(f"Métricas de Avaliação (Holdout Final): {self.nome_modelo}")
        print(f"{'='*60}")
        display(df_metricas.round(4))
        
        return df_metricas

    def plotar_curva_roc(self) -> None:
        """
        Plota a Curva ROC comparando Treino e Holdout.
        """
        fpr_train, tpr_train, _ = roc_curve(self.y_train, self.y_prob_train)
        fpr_test, tpr_test, _ = roc_curve(self.y_holdout, self.y_prob_holdout)
        
        auc_train = roc_auc_score(self.y_train, self.y_prob_train)
        auc_test = roc_auc_score(self.y_holdout, self.y_prob_holdout)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_train, tpr_train, label=f'Treino (AUC = {auc_train:.3f})', color='blue', linestyle='--')
        plt.plot(fpr_test, tpr_test, label=f'Holdout (AUC = {auc_test:.3f})', color='darkred', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k:', label='Chute Aleatório (AUC = 0.500)')
        
        plt.title(f'Curva ROC - {self.nome_modelo}', fontsize=14)
        plt.xlabel('Taxa de Falso Positivo (FPR)')
        plt.ylabel('Taxa de Verdadeiro Positivo (TPR)')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plotar_curva_lift(self, n_bins: int = 10) -> None:
        """
        Plota o Lift Chart (Curva de Ganho Acumulado).
        Mostra quantas vezes o modelo é melhor que escolher clientes aleatoriamente.
        """
        # Criando DataFrame de ordenação (Usa o HOLDOUT!)
        df_lift = pd.DataFrame({'y_true': self.y_holdout, 'y_prob': self.y_prob_holdout})
        df_lift = df_lift.sort_values(by='y_prob', ascending=False).reset_index(drop=True)
        
        # Calculando as métricas acumuladas
        df_lift['random_prob'] = df_lift['y_true'].mean()
        df_lift['cumulative_churn'] = df_lift['y_true'].cumsum()
        df_lift['population_pct'] = (df_lift.index + 1) / len(df_lift)
        
        # Lift = (Churn acumulado / Total de churns) / População testada
        total_churns = df_lift['y_true'].sum()
        df_lift['lift'] = (df_lift['cumulative_churn'] / total_churns) / df_lift['population_pct']
        
        # Plotando
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plotando a linha de base aleatória (Lift = 1)
        ax.plot([0, 1], [1, 1], 'k--', label='Aleatório (Lift = 1)')
        
        # Plotando o Lift do modelo
        ax.plot(df_lift['population_pct'], df_lift['lift'], color='darkblue', linewidth=2, label='Modelo')
        
        # Adicionando marcadores para 10%, 20% e 30% da população
        for pct in [0.1, 0.2, 0.3]:
            idx = int(len(df_lift) * pct) - 1
            lift_val = df_lift.loc[idx, 'lift']
            ax.scatter(pct, lift_val, color='darkred', s=100, zorder=5)
            ax.annotate(f'Lift {lift_val:.2f}x', (pct, lift_val), textcoords="offset points", xytext=(0,10), ha='center', color='darkred', fontweight='bold')
            
        ax.set_title(f'Curva de Lift (Ganho Acumulado) - {self.nome_modelo} (Holdout)', fontsize=14)
        ax.set_xlabel('Percentual da População (Ordenada por Risco)')
        ax.set_ylabel('Lift (Vezes melhor que o aleatório)')
        ax.set_xlim(0, 1)
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plotar_matriz_confusao(self) -> None:
        """
        Plota a matriz de confusão do conjunto de Holdout.
        """
        cm = confusion_matrix(self.y_holdout, self.y_pred_holdout)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ficou (0)', 'Churn (1)'])
        
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title(f'Matriz de Confusão (Holdout) - {self.nome_modelo}', fontsize=14)
        plt.tight_layout()
        plt.show()

    def gerar_relatorio(self) -> None:
        """
        Roda tudo de uma vez.
        """
        self.calcular_metricas()
        self.plotar_curva_roc()
        self.plotar_curva_lift()
        self.plotar_matriz_confusao()