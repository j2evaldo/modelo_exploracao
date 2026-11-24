import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

sns.set()

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
        df_metricas['Gap (Treino - Holdout)'] = df_metricas['Treino'] - df_metricas['Holdout (Teste)']
        
        print(f"{'='*60}")
        print(f"Métricas de Avaliação (Holdout Final): {self.nome_modelo}")
        print(f"{'='*60}")
        display(df_metricas.round(4))
        
        return df_metricas

    def plotar_curva_roc(self):
        """
        Plota a Curva ROC. Retorna a figura para o MLflow.
        """
        fpr_train, tpr_train, _ = roc_curve(self.y_train, self.y_prob_train)
        fpr_test, tpr_test, _ = roc_curve(self.y_holdout, self.y_prob_holdout)
        
        auc_train = roc_auc_score(self.y_train, self.y_prob_train)
        auc_test = roc_auc_score(self.y_holdout, self.y_prob_holdout)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr_train, tpr_train, label=f'Treino (AUC = {auc_train:.3f})', color='blue', linestyle='--')
        ax.plot(fpr_test, tpr_test, label=f'Holdout (AUC = {auc_test:.3f})', color='darkred', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k:', label='Chute Aleatório (AUC = 0.500)')
        
        ax.set_title(f'Curva ROC - {self.nome_modelo}', fontsize=14)
        ax.set_xlabel('Taxa de Falso Positivo (FPR)')
        ax.set_ylabel('Taxa de Verdadeiro Positivo (TPR)')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.close(fig)
        return fig

    def plotar_curva_lift(self):
        """
        Plota a Curva de Lift (Ganho Acumulado). Retorna a figura para o MLflow.
        """
        df_lift = pd.DataFrame({'y_true': self.y_holdout, 'y_prob': self.y_prob_holdout})
        df_lift = df_lift.sort_values(by='y_prob', ascending=False).reset_index(drop=True)
        
        df_lift['cumulative_churn'] = df_lift['y_true'].cumsum()
        df_lift['population_pct'] = (df_lift.index + 1) / len(df_lift)
        
        total_churns = df_lift['y_true'].sum()
        df_lift['cumulative_gain'] = df_lift['cumulative_churn'] / total_churns
        df_lift['lift'] = df_lift['cumulative_gain'] / df_lift['population_pct']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot([0, 1], [0, 1], 'k--', label='Aleatório (Lift = 1)')
        ax.plot(df_lift['population_pct'], df_lift['cumulative_gain'], color='darkblue', linewidth=2, label='Modelo (Ganho Acumulado)')
        
        for pct in [0.1, 0.2, 0.3]:
            idx = int(len(df_lift) * pct) - 1
            gain_val = df_lift.loc[idx, 'cumulative_gain']
            lift_val = df_lift.loc[idx, 'lift']
            
            ax.scatter(pct, gain_val, color='darkred', s=100, zorder=5)
            ax.annotate(f'{gain_val*100:.1f}% capturado\n(Lift {lift_val:.2f}x)', 
                        (pct, gain_val), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center', 
                        color='darkred', 
                        fontweight='bold')
            
        ax.set_title(f'Curva de Lift (Ganho Acumulado) - {self.nome_modelo} (Holdout)', fontsize=14)
        ax.set_xlabel('Percentual da População (Ordenada por Risco)')
        ax.set_ylabel('Percentual de Churns Capturados')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.close(fig)
        return fig

    def plotar_matriz_confusao(self):
        """
        Plota a matriz de confusão. Retorna a figura para o MLflow.
        """
        cm = confusion_matrix(self.y_holdout, self.y_pred_holdout)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ficou (0)', 'Churn (1)'])
        
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title(f'Matriz de Confusão (Holdout) - {self.nome_modelo}', fontsize=14)
        
        plt.tight_layout()
        plt.close(fig)
        return fig

    def plotar_feature_importance(self, top_n: int = 15, usar_shap: bool = False):
        """
        Plota a importância das features. Retorna a figura para o MLflow.
        """
        if usar_shap:
            if hasattr(self.modelo, 'get_feature_importance'):
                shap_values = self.modelo.get_feature_importance(type='ShapValues', data=self.X_holdout)
                importances = np.abs(shap_values[:, :-1]).mean(axis=0)
            else:
                raise ValueError("O modelo não suporta SHAP nativo. Use usar_shap=False.")
        else:
            if hasattr(self.modelo, 'get_feature_importance'):
                importances = self.modelo.get_feature_importance()
            elif hasattr(self.modelo, 'feature_importances_'):
                importances = self.modelo.feature_importances_
            else:
                raise ValueError("O modelo não possui atributo de importância de features.")

        df_importance = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
        
        sns.barplot(
            data=df_importance, 
            x='Importance', 
            y='Feature', 
            palette='viridis',
            hue='Feature',
            legend=False,
            ax=ax
        )
        
        metodo = "SHAP" if usar_shap else "Nativa"
        ax.set_title(f'Feature Importance ({metodo}) - {self.nome_modelo}', fontsize=14)
        ax.set_xlabel('Importância')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        plt.close(fig)
        return fig

    def gerar_relatorio(self):
        """
        Roda tudo de uma vez exibindo cada figura uma única vez.
        """
        self.calcular_metricas()

        display(self.plotar_curva_roc())
        display(self.plotar_curva_lift())
        display(self.plotar_matriz_confusao())
        display(self.plotar_feature_importance())