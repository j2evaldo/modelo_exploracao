import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy.stats import chi2_contingency, mannwhitneyu

import warnings
warnings.filterwarnings('ignore')


class AnaliseBivariada:
    def __init__(self, df: pd.DataFrame, coluna_alvo: str):
        self.df = df
        self.alvo = coluna_alvo

    def analise_numerica_vs_alvo(self, coluna: str) -> None:
        """
        Analisa uma variável numérica contra o alvo (binário).
        Gera Boxplot, KDE sobreposto, tabela de médias e Teste de Mann-Whitney.
        """
        print(f"{'='*60}")
        print(f"Análise Bivariada (Numérica vs Alvo): {coluna} x {self.alvo}")
        print(f"{'='*60}")
        
        # 1. Estatísticas descritivas agrupadas
        resumo = self.df.groupby(self.alvo)[coluna].agg(['mean', 'median', 'std']).round(2)
        display(resumo)
        
        # 2. Teste estatístico (Mann-Whitney U, pois não assume normalidade)
        # Separa os grupos com base no alvo (ex: 0 e 1)
        grupos = self.df[self.alvo].unique()
        if len(grupos) == 2:
            grupo_0 = self.df[self.df[self.alvo] == grupos[0]][coluna].dropna()
            grupo_1 = self.df[self.df[self.alvo] == grupos[1]][coluna].dropna()
            stat, p_value = mannwhitneyu(grupo_0, grupo_1)
            print(f"Teste de Mann-Whitney U - p-value: {p_value:.4e}")
            if p_value < 0.05:
                print(">> Diferença estatisticamente significativa entre os grupos.\n")
            else:
                print(">> Sem diferença estatística significativa entre os grupos.\n")
        
        # 3. Gráficos Lado a Lado
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Boxplot
        sns.boxplot(
            data=self.df, 
            x=self.alvo, 
            y=coluna, 
            ax=axes[0], 
            palette="Set2",
            hue=self.alvo,
            legend=False
        )
        axes[0].set_title(f"{coluna} por {self.alvo} (Boxplot)")
        
        # KDE Plot (Densidade)
        sns.kdeplot(
            data=self.df, 
            x=coluna, 
            hue=self.alvo, 
            common_norm=False, 
            fill=True, 
            ax=axes[1],
            palette="Set2"
        )
        axes[1].set_title(f"Densidade de {coluna} por {self.alvo}")
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def analise_categorica_vs_alvo(self, coluna: str) -> None:
        """
        Analisa uma variável categórica contra o alvo.
        Gera Tabela de contingência (%), Teste Qui-Quadrado e Gráfico de Barras Empilhadas.
        """
        print(f"{'='*60}")
        print(f"Análise Bivariada (Categórica vs Alvo): {coluna} x {self.alvo}")
        print(f"{'='*60}")
        
        # 1. Tabela de contingência com proporções (taxa do alvo por categoria)
        # normalize='index' calcula a proporção em relação ao total da linha
        crosstab = pd.crosstab(self.df[coluna], self.df[self.alvo], normalize='index') * 100
        crosstab = crosstab.round(2)
        crosstab.columns = [f"{col}_%" for col in crosstab.columns]
        display(crosstab)
        
        # 2. Teste Qui-Quadrado
        tabela_contingencia = pd.crosstab(self.df[coluna], self.df[self.alvo])
        chi2, p_value, dof, expected = chi2_contingency(tabela_contingencia)
        print(f"Teste Qui-Quadrado - p-value: {p_value:.4e}")
        if p_value < 0.05:
            print(f">> A variável {coluna} tem associação significativa com o alvo.\n")
        else:
            print(f">> A variável {coluna} NÃO tem associação significativa com o alvo.\n")

        # 3. Gráfico de Barras Empilhadas 100%
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Criando um dataframe temporário com proporções para o gráfico
        df_plot = pd.crosstab(self.df[coluna], self.df[self.alvo], normalize='index')
        
        df_plot.plot(
            kind='barh', 
            stacked=True, 
            ax=ax, 
            color=sns.color_palette("Set2"),
            figsize=(10, 5)
        )
        
        ax.set_title(f"Proporção de {self.alvo} por {coluna}")
        ax.set_xlabel("Proporção")
        ax.set_ylabel(coluna)
        ax.legend(title=self.alvo, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
    def heatmap_interacao(self, coluna1: str, coluna2: str) -> None:
        """
        Bônus: Analisa a interação de duas variáveis categóricas 
        e como a taxa do alvo se comporta.
        """
        print(f"{'='*60}")
        print(f"Interação: {coluna1} x {coluna2} (Taxa de {self.alvo})")
        print(f"{'='*60}")
        
        # Calcula a média do alvo (se for 0/1, vira a taxa percentual)
        pivot = self.df.pivot_table(
            values=self.alvo, 
            index=coluna1, 
            columns=coluna2, 
            aggfunc='mean'
        ) * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            pivot, 
            annot=True, 
            fmt=".1f", 
            cmap="Reds", 
            ax=ax,
            cbar_kws={'label': f'Taxa de {self.alvo} (%)'}
        )
        ax.set_title(f"Taxa de {self.alvo} por {coluna1} e {coluna2}")
        plt.tight_layout()
        plt.show()
        plt.close(fig)