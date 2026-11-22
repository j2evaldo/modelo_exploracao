import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import warnings
warnings.filterwarnings('ignore')

class Exploratoria:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def analise_univariada_numerica(self, coluna: str) -> None:
        """
        Gera um describe e uma figura com 3 painéis:
        1. Displot (Histograma + Densidade)
        2. Boxplot Clássico
        3. Boxplot com Pontos (Strip Plot sobreposto)
        """
        print(f"{'='*50}")
        print(f"Estatísticas Descritivas: {coluna}")
        print(f"{'='*50}")
        
        # 1. Tabela de describe
        desc = self.df[coluna].describe().to_frame().T
        display(desc)
        
        # 2. Configura a figura
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # --- Gráfico 1: Displot (Histograma + KDE) ---
        sns.histplot(
            data=self.df, 
            x=coluna, 
            kde=True, 
            ax=axes[0], 
            color="skyblue",
            bins=30
        )
        axes[0].set_title(f"Distribuição (Displot) - {coluna}")
        axes[0].set_ylabel("Frequência")
        
        # --- Gráfico 2: Boxplot Clássico ---
        sns.boxplot(
            data=self.df, 
            y=coluna, 
            ax=axes[1], 
            color="lightgreen",
            width=0.4
        )
        axes[1].set_title(f"Boxplot Clássico - {coluna}")
        axes[1].set_xlabel(coluna)
        
        # --- Gráfico 3: Boxplot com Pontos ---
        sns.boxplot(
            data=self.df, 
            y=coluna, 
            ax=axes[2], 
            color="white", 
            width=0.5,
            boxprops=dict(alpha=0.7)
        )
        
        sns.stripplot(
            data=self.df, 
            y=coluna, 
            ax=axes[2], 
            color="coral", 
            alpha=0.3, 
            jitter=0.15, 
            size=4
        )
        axes[2].set_title(f"Boxplot + Concentração - {coluna}")
        axes[2].set_xlabel(coluna)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)  # Fecha a figura para liberar memória

    def analise_univariada_categorica(self, coluna: str, usar_pizza: bool = False) -> None:
        """
        Gera tabela de frequência e gráfico de barras (ou pizza) para uma coluna categórica.
        """
        print(f"{'='*50}")
        print(f"Frequência: {coluna}")
        print(f"{'='*50}")
        
        # 1. Tabela de frequência absoluta e relativa
        freq = self.df[coluna].value_counts().to_frame("contagem")
        freq["proporcao_%"] = (self.df[coluna].value_counts(normalize=True) * 100).round(2)
        display(freq)
        
        # 2. Configura a figura
        if usar_pizza:
            fig, ax = plt.subplots(figsize=(6, 6))
            self.df[coluna].value_counts().plot.pie(
                autopct='%1.1f%%', 
                ax=ax, 
                colors=sns.color_palette("pastel"),
                startangle=90
            )
            ax.set_ylabel("") 
            ax.set_title(f"Proporção de {coluna}")
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(
                data=self.df, 
                y=coluna, 
                hue=coluna,             # Correção do FutureWarning
                legend=False,           # Correção do FutureWarning
                order=self.df[coluna].value_counts().index, 
                palette="viridis",
                ax=ax
            )
            ax.set_title(f"Distribuição de {coluna}")
            ax.set_xlabel("Contagem")
            ax.set_ylabel(coluna)
            
            # Adiciona os valores numéricos no final de cada barra
            for p in ax.patches:
                ax.annotate(f'{int(p.get_width())}', 
                            (p.get_width(), p.get_y() + p.get_height() / 2.), 
                            ha='left', va='center', 
                            xytext=(5, 0), textcoords='offset points')
                
        plt.tight_layout()
        plt.show()
        plt.close(fig)  # Fecha a figura para liberar memória

    def analise_univariada(self, coluna: str, usar_pizza: bool = False) -> None:
        """
        Roteador automático: identifica se a coluna é numérica ou categórica
        e chama o método de análise apropriado.
        """
        if pd.api.types.is_numeric_dtype(self.df[coluna]):
            self.analise_univariada_numerica(coluna)
        else:
            self.analise_univariada_categorica(coluna, usar_pizza=usar_pizza)