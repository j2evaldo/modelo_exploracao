import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal

class AnaliseBivariada:
    def __init__(self, df: pd.DataFrame, coluna_alvo: str):
        self.df = df
        self.alvo = coluna_alvo

    def analise_numerica_vs_alvo(self, coluna: str) -> None:
        """
        Analisa uma variável numérica contra o alvo (binário).
        Gera Boxplot (com pontos de densidade), KDE sobreposto, tabela de 
        médias (com min/max) e Teste de Mann-Whitney.
        """
        print(f"{'='*60}")
        print(f"Análise Bivariada (Numérica vs Alvo): {coluna} x {self.alvo}")
        print(f"{'='*60}")
        
        # 1. Estatísticas descritivas agrupadas (com min e max)
        resumo = self.df.groupby(self.alvo)[coluna].agg(
            ['count', 'mean', 'median', 'std', 'min', 'max']
        ).round(2)
        display(resumo)
        
        # 2. Verificação de limites e possíveis outliers extremos
        valor_min = self.df[coluna].min()
        valor_max = self.df[coluna].max()
        
        print(f"Verificação de Limites para {coluna}:")
        print(f"  - Mínimo global: {valor_min}")
        print(f"  - Máximo global: {valor_max}")
        
        if valor_min < 0:
            print(f"  [ALERTA] Foram encontrados valores negativos em '{coluna}'.")
            qtd_negativos = (self.df[coluna] < 0).sum()
            print(f"  Existem {qtd_negativos} registros com valor negativo.")
            
        if coluna == 'tenure' and valor_max > 72:
            print(f"  [ALERTA] Valor máximo de tenure ({valor_max}) maior que o esperado (72 meses).")
        elif coluna == 'MonthlyCharges' and valor_max > 150:
            print(f"  [ALERTA] Mensalidade máxima ({valor_max}) parece muito alta.")
        print("")

        # 3. Teste estatístico (Mann-Whitney U)
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
        
        # 4. Gráficos Lado a Lado
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Boxplot com Pontos sobrepostos
        sns.boxplot(
            data=self.df, 
            x=self.alvo, 
            y=coluna, 
            ax=axes[0], 
            palette="Set2",
            hue=self.alvo,
            legend=False,
            width=0.4,
            boxprops=dict(alpha=0.7)
        )
        sns.stripplot(
            data=self.df,
            x=self.alvo,
            y=coluna,
            ax=axes[0],
            color="black",
            alpha=0.2,
            size=2,
            jitter=0.15
        )
        axes[0].set_title(f"{coluna} por {self.alvo} (Boxplot + Densidade)")
        
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
        Gera Tabela de contingência (%), Tabela de desvio, Teste Qui-Quadrado 
        e Gráfico de Barras Empilhadas com linha de média global do alvo.
        """
        print(f"{'='*60}")
        print(f"Análise Bivariada (Categórica vs Alvo): {coluna} x {self.alvo}")
        print(f"{'='*60}")
        
        # 1. Tabela de contingência com proporções
        crosstab = pd.crosstab(self.df[coluna], self.df[self.alvo], normalize='index') * 100
        crosstab = crosstab.round(2)
        crosstab.columns = [f"{col}_%" for col in crosstab.columns]
        display(crosstab)
        
        # 2. Tabela de Desvio em relação à Média Global do Alvo
        taxa_global = self.df[self.alvo].mean() * 100
        coluna_churn_1 = crosstab.columns[-1]
        desvio = crosstab[coluna_churn_1] - taxa_global
        
        print(f"Média Global de {self.alvo}: {taxa_global:.2f}%")
        print(f"Desvio de cada categoria em relação à Média Global (pontos percentuais):")
        display(desvio.round(2).to_frame("Desvio (pp)"))
        
        # 3. Teste Qui-Quadrado
        tabela_contingencia = pd.crosstab(self.df[coluna], self.df[self.alvo])
        chi2, p_value, dof, expected = chi2_contingency(tabela_contingencia)
        print(f"Teste Qui-Quadrado - p-value: {p_value:.4e}")
        if p_value < 0.05:
            print(f">> A variável {coluna} tem associação significativa com o alvo.\n")
        else:
            print(f">> A variável {coluna} NÃO tem associação significativa com o alvo.\n")

        # 4. Gráfico de Barras Empilhadas 100%
        fig, ax = plt.subplots(figsize=(10, 5))
        
        df_plot = pd.crosstab(self.df[coluna], self.df[self.alvo], normalize='index')
        
        df_plot.plot(
            kind='barh', 
            stacked=True, 
            ax=ax, 
            color=sns.color_palette("Set2"),
            figsize=(10, 5)
        )
        
        # Linha de referência da média global do ALVO
        ax.axvline(
            x=self.df[self.alvo].mean(), 
            color='darkred', 
            linestyle='--', 
            linewidth=2,
            label=f'Média Global ({taxa_global:.1f}%)'
        )
        
        # Texto da média perto da linha
        ax.text(
            self.df[self.alvo].mean() + 0.01, 
            len(df_plot) - 0.5, 
            f'{taxa_global:.1f}%', 
            color='darkred', 
            fontweight='bold',
            va='center'
        )
        
        ax.set_title(f"Proporção de {self.alvo} por {coluna}")
        ax.set_xlabel("Proporção")
        ax.set_ylabel(coluna)
        
        # Atualizando a legenda para incluir a linha de média
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, title=self.alvo, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
    def analise_numerica_vs_categorica(self, coluna_num: str, coluna_cat: str) -> None:
        """
        Compara uma variável numérica entre grupos de uma categórica.
        Plota boxplot com a linha da média global da FEATURE (não do alvo)
        e exibe uma tabela de desvios.
        """
        print(f"{'='*60}")
        print(f"Análise: {coluna_num} por {coluna_cat}")
        print(f"{'='*60}")
        
        # 1. Estatísticas da feature por categoria
        resumo = self.df.groupby(coluna_cat)[coluna_num].agg(
            ['count', 'mean', 'median', 'min', 'max']
        ).round(2)
        
        # Calculando a média global da FEATURE (não do alvo)
        media_global_feature = self.df[coluna_num].mean()
        
        # Adicionando o desvio em relação à média global da feature
        resumo['desvio_vs_global'] = (resumo['mean'] - media_global_feature).round(2)
        
        print(f"Média Global de {coluna_num}: {media_global_feature:.2f}")
        print(f"Resumo da feature por categoria de {coluna_cat}:")
        display(resumo)
        
        # 2. Boxplot com a linha da média global da feature
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.boxplot(
            data=self.df, 
            x=coluna_cat, 
            y=coluna_num, 
            palette="Set2", 
            ax=ax, 
            hue=coluna_cat, 
            legend=False
        )
        
        # Linha horizontal com a média global da FEATURE
        ax.axhline(
            y=media_global_feature, 
            color='darkred', 
            linestyle='--', 
            linewidth=2, 
            label=f'Média Global de {coluna_num} ({media_global_feature:.1f})'
        )
        
        ax.set_title(f"Distribuição de {coluna_num} por {coluna_cat}")
        ax.set_ylabel(coluna_num)
        ax.set_xlabel(coluna_cat)
        plt.xticks(rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        # 3. Teste de Kruskal-Wallis
        grupos = [grupo[coluna_num].dropna().values 
                  for _, grupo in self.df.groupby(coluna_cat)]
        stat, p_value = kruskal(*grupos)
        print(f"Teste de Kruskal-Wallis - p-value: {p_value:.4e}")
        if p_value < 0.05:
            print(f">> A variável {coluna_num} difere significativamente entre as categorias de {coluna_cat}.\n")
        else:
            print(f">> Sem diferença significativa entre as categorias.\n")

    def analise_binning_vs_alvo(self, coluna: str, bins: int = 10) -> None:
        """
        Divide uma variável numérica em faixas (bins) e mostra a taxa do alvo em cada faixa.
        Excelente para capturar relações não-lineares.
        """
        print(f"{'='*60}")
        print(f"Taxa de {self.alvo} por faixas de {coluna}")
        print(f"{'='*60}")
        
        # Criando as faixas
        faixas = pd.cut(self.df[coluna], bins=bins, duplicates='drop')
        
        # Calculando a taxa do alvo por faixa
        taxa_por_faixa = self.df.groupby(faixas, observed=True)[self.alvo].agg(['mean', 'count'])
        taxa_por_faixa['mean'] = taxa_por_faixa['mean'] * 100 # Convertendo para %
        display(taxa_por_faixa.round(2))
        
        # Plotando
        fig, ax = plt.subplots(figsize=(12, 5))
        taxa_por_faixa['mean'].plot(kind='line', marker='o', color='darkblue', ax=ax)
        
        # Linha da média global do alvo
        taxa_global = self.df[self.alvo].mean() * 100
        ax.axhline(y=taxa_global, color='darkred', linestyle='--', label=f'Média Global ({taxa_global:.1f}%)')
        
        ax.set_title(f"Taxa de {self.alvo} por faixas de {coluna}")
        ax.set_ylabel(f"Taxa de {self.alvo} (%)")
        ax.set_xlabel(f"Faixas de {coluna}")
        plt.xticks(rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
            
    def heatmap_interacao(self, coluna1: str, coluna2: str) -> None:
        """
        Analisa a interação de duas variáveis categóricas 
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
        
        fig, ax = plt.subplots(figsize=(12, 10))
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

    def matriz_correlacao_numerica(self, metodo: str = 'spearman') -> None:
        """
        Gera a matriz de correlação das variáveis numéricas.
        O método 'spearman' é mais robusto para dados não-normais e ordinais.
        """
        print(f"{'='*60}")
        print(f"Matriz de Correlação ({metodo.capitalize()}) - Numéricas")
        print(f"{'='*60}")
        
        # Seleciona apenas colunas numéricas
        df_num = self.df.select_dtypes(include=['number'])
        
        if df_num.empty:
            print("Nenhuma coluna numérica encontrada para calcular a correlação.")
            return
            
        # Calcula a correlação
        corr = df_num.corr(method=metodo)
        
        # Configura a máscara para esconder a parte de cima (triângulo superior)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            corr, 
            mask=mask, 
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm", 
            vmin=-1, vmax=1, 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            ax=ax
        )
        
        # Destacando a coluna do alvo para facilitar a leitura
        if self.alvo in corr.columns:
            idx_alvo = list(corr.columns).index(self.alvo)
            ax.add_patch(plt.Rectangle((idx_alvo, 0), 1, len(corr), fill=False, edgecolor='red', lw=2))
            
        ax.set_title(f"Matriz de Correlação ({metodo.capitalize()})", fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def _calcular_cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """
        Função auxiliar (privada) para calcular o V de Cramér entre duas séries.
        """
        tabela = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(tabela)
        n_obs = tabela.sum().sum()
        
        if n_obs == 0:
            return 0.0
            
        phi2 = chi2 / n_obs
        r, k = tabela.shape
        
        # Correção de viés
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n_obs-1))
        rcorr = r - ((r-1)**2)/(n_obs-1)
        kcorr = k - ((k-1)**2)/(n_obs-1)
        
        denominador = min((kcorr-1), (rcorr-1))
        if denominador <= 0:
            return 0.0
            
        return np.sqrt(phi2corr / denominador)

    def matriz_correlacao_categorica(self) -> None:
        """
        Calcula a associação entre variáveis categóricas usando o V de Cramér.
        Gera um heatmap que varia de 0 (sem associação) a 1 (associação perfeita).
        """
        print(f"{'='*60}")
        print("Matriz de Correlação (Cramér's V) - Categóricas")
        print(f"{'='*60}")
        
        # Seleciona apenas colunas categóricas (strings)
        df_cat = self.df.select_dtypes(include=['object', 'category'])
        
        # Remove o alvo da matriz (se ele for categórico) para focar nas features
        if self.alvo in df_cat.columns:
            df_cat = df_cat.drop(columns=[self.alvo])
            
        if df_cat.empty:
            print("Nenhuma coluna categórica encontrada.")
            return

        colunas = df_cat.columns
        n = len(colunas)
        
        # Inicializa matriz vazia
        cramers_v = np.zeros((n, n))
        
        # Calcula par a par
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    cramers_v[i, j] = 1.0
                else:
                    val = self._calcular_cramers_v(df_cat[colunas[i]], df_cat[colunas[j]])
                    cramers_v[i, j] = val
                    cramers_v[j, i] = val  # Matriz simétrica

        # Criando DataFrame para o heatmap
        df_cramers = pd.DataFrame(cramers_v, index=colunas, columns=colunas)
        
        # Plotando
        mask = np.triu(np.ones_like(df_cramers, dtype=bool))
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            df_cramers, 
            mask=mask, 
            annot=True, 
            fmt=".2f", 
            cmap="YlOrRd", 
            vmin=0, vmax=1, 
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            ax=ax
        )
        
        ax.set_title("Matriz de Associação entre Categóricas (Cramér's V)", fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.close(fig)