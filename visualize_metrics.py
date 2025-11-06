"""
Script para visualizar as métricas coletadas durante a execução do algoritmo genético.

Este script lê os arquivos generation_history.csv e gera gráficos detalhados
mostrando todas as métricas conforme documentado em METRICS_DOCUMENTATION.md
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

# Configurar estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def plot_convergence(df, output_path=None):
    """
    Plota a convergência do fitness ao longo das gerações.
    
    Mostra:
    - Melhor fitness observado (overall)
    - Melhor fitness da geração
    - Fitness médio da população
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['generation'], df['best_fitness_overall'], 
            label='Melhor Observado (Overall)', linewidth=2.5, color='#2E86AB')
    ax.plot(df['generation'], df['best_fitness_generation'], 
            label='Melhor da Geração', linewidth=1.5, alpha=0.8, color='#A23B72')
    ax.plot(df['generation'], df['avg_fitness'], 
            label='Fitness Médio', linestyle='--', linewidth=1.5, color='#F18F01')
    
    # Preencher área entre melhor e médio
    ax.fill_between(df['generation'], df['best_fitness_overall'], df['avg_fitness'], 
                     alpha=0.2, color='#2E86AB')
    
    ax.set_xlabel('Geração', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fitness', fontsize=12, fontweight='bold')
    ax.set_title('Convergência do Algoritmo Genético\n(Fitness ao longo das Gerações)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_diversity_vs_fitness(df, output_path=None):
    """
    Plota fitness e diversidade no mesmo gráfico (eixos Y duplos).
    
    Mostra a relação entre convergência e perda de diversidade.
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Eixo 1: Fitness
    color1 = '#2E86AB'
    ax1.set_xlabel('Geração', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fitness', color=color1, fontsize=12, fontweight='bold')
    ax1.plot(df['generation'], df['best_fitness_overall'], 
             color=color1, linewidth=2.5, label='Best Fitness')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Eixo 2: Diversidade
    ax2 = ax1.twinx()
    color2 = '#E63946'
    ax2.set_ylabel('Diversidade da População', color=color2, fontsize=12, fontweight='bold')
    ax2.plot(df['generation'], df['diversity'], 
             color=color2, linewidth=2, linestyle='--', label='Diversity')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Título
    plt.title('Fitness vs. Diversidade da População\n(Trade-off Exploração vs. Exploração)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Legendas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=11)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_fitness_range(df, output_path=None):
    """
    Plota o range de fitness (min, avg, max) ao longo das gerações.
    
    Mostra como a população se torna mais homogênea.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Preencher área entre min e max
    ax.fill_between(df['generation'], df['min_fitness'], df['max_fitness'], 
                     alpha=0.3, color='#A8DADC', label='Range (Min-Max)')
    
    ax.plot(df['generation'], df['max_fitness'], 
            label='Fitness Máximo', linewidth=2, color='#457B9D')
    ax.plot(df['generation'], df['avg_fitness'], 
            label='Fitness Médio', linewidth=2, color='#1D3557')
    ax.plot(df['generation'], df['min_fitness'], 
            label='Fitness Mínimo', linewidth=2, color='#E63946')
    
    ax.set_xlabel('Geração', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fitness', fontsize=12, fontweight='bold')
    ax.set_title('Range de Fitness da População\n(Homogeneização ao longo da Evolução)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_improvement_rate(df, output_path=None):
    """
    Plota a taxa de melhoria (derivada) do fitness ao longo das gerações.
    
    Mostra quando o algoritmo está melhorando mais rapidamente.
    """
    # Calcular taxa de melhoria (diferença entre gerações consecutivas)
    df['improvement_rate'] = df['best_fitness_overall'].diff().fillna(0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Subplot 1: Fitness
    ax1.plot(df['generation'], df['best_fitness_overall'], 
             linewidth=2.5, color='#2E86AB')
    ax1.set_ylabel('Fitness', fontsize=12, fontweight='bold')
    ax1.set_title('Fitness e Taxa de Melhoria', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Taxa de melhoria
    ax2.bar(df['generation'], df['improvement_rate'], 
            color='#F18F01', alpha=0.7, edgecolor='#C75000')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Geração', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Taxa de Melhoria\n(Δ Fitness)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_all_metrics_dashboard(df, output_path=None):
    """
    Cria um dashboard completo com todas as métricas.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # 1. Convergência do Fitness
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['generation'], df['best_fitness_overall'], 
             label='Melhor Overall', linewidth=2.5, color='#2E86AB')
    ax1.plot(df['generation'], df['best_fitness_generation'], 
             label='Melhor da Geração', linewidth=1.5, alpha=0.8, color='#A23B72')
    ax1.plot(df['generation'], df['avg_fitness'], 
             label='Médio', linestyle='--', linewidth=1.5, color='#F18F01')
    ax1.fill_between(df['generation'], df['best_fitness_overall'], df['avg_fitness'], 
                     alpha=0.2, color='#2E86AB')
    ax1.set_ylabel('Fitness', fontweight='bold')
    ax1.set_title('Convergência do Fitness', fontweight='bold', pad=10)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Range de Fitness
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(df['generation'], df['min_fitness'], df['max_fitness'], 
                     alpha=0.3, color='#A8DADC')
    ax2.plot(df['generation'], df['max_fitness'], linewidth=1.5, color='#457B9D')
    ax2.plot(df['generation'], df['avg_fitness'], linewidth=1.5, color='#1D3557')
    ax2.plot(df['generation'], df['min_fitness'], linewidth=1.5, color='#E63946')
    ax2.set_ylabel('Fitness', fontweight='bold')
    ax2.set_title('Range de Fitness', fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Diversidade
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df['generation'], df['diversity'], 
             linewidth=2, color='#E63946')
    ax3.fill_between(df['generation'], 0, df['diversity'], alpha=0.3, color='#E63946')
    ax3.set_ylabel('Diversidade', fontweight='bold')
    ax3.set_title('Diversidade da População', fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Taxa de Melhoria
    df['improvement_rate'] = df['best_fitness_overall'].diff().fillna(0)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.bar(df['generation'], df['improvement_rate'], 
            color='#F18F01', alpha=0.7, edgecolor='#C75000')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Geração', fontweight='bold')
    ax4.set_ylabel('Δ Fitness', fontweight='bold')
    ax4.set_title('Taxa de Melhoria', fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Estatísticas
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Calcular estatísticas
    stats_text = f"""
    ESTATÍSTICAS RESUMIDAS
    
    Fitness Inicial: {df['best_fitness_overall'].iloc[0]:.4f}
    Fitness Final: {df['best_fitness_overall'].iloc[-1]:.4f}
    Melhoria Total: {df['best_fitness_overall'].iloc[-1] - df['best_fitness_overall'].iloc[0]:.4f}
    Melhoria %: {((df['best_fitness_overall'].iloc[-1] - df['best_fitness_overall'].iloc[0]) / abs(df['best_fitness_overall'].iloc[0]) * 100):.2f}%
    
    Diversidade Inicial: {df['diversity'].iloc[0]:.4f}
    Diversidade Final: {df['diversity'].iloc[-1]:.4f}
    Redução Diversidade: {((df['diversity'].iloc[0] - df['diversity'].iloc[-1]) / df['diversity'].iloc[0] * 100):.1f}%
    
    Fitness Médio Inicial: {df['avg_fitness'].iloc[0]:.4f}
    Fitness Médio Final: {df['avg_fitness'].iloc[-1]:.4f}
    
    Total de Gerações: {len(df)}
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('Dashboard de Métricas do Algoritmo Genético', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard salvo: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualizar métricas do Algoritmo Genético')
    parser.add_argument('history_file', type=str, 
                       help='Caminho para o arquivo generation_history.csv')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Diretório para salvar os gráficos (None = mostrar na tela)')
    parser.add_argument('--dashboard-only', action='store_true',
                       help='Gerar apenas o dashboard completo')
    
    args = parser.parse_args()
    
    # Verificar se o arquivo existe
    if not os.path.exists(args.history_file):
        print(f"Erro: Arquivo não encontrado: {args.history_file}")
        return
    
    # Carregar dados
    print(f"Carregando dados de: {args.history_file}")
    df = pd.read_csv(args.history_file)
    
    # Criar diretório de saída se especificado
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Gráficos serão salvos em: {args.output_dir}")
    
    # Gerar gráficos
    if args.dashboard_only:
        output_path = os.path.join(args.output_dir, 'dashboard.png') if args.output_dir else None
        plot_all_metrics_dashboard(df, output_path)
    else:
        # Convergência
        output_path = os.path.join(args.output_dir, 'convergence.png') if args.output_dir else None
        plot_convergence(df, output_path)
        
        # Diversidade vs Fitness
        output_path = os.path.join(args.output_dir, 'diversity_vs_fitness.png') if args.output_dir else None
        plot_diversity_vs_fitness(df, output_path)
        
        # Range de Fitness
        output_path = os.path.join(args.output_dir, 'fitness_range.png') if args.output_dir else None
        plot_fitness_range(df, output_path)
        
        # Taxa de Melhoria
        output_path = os.path.join(args.output_dir, 'improvement_rate.png') if args.output_dir else None
        plot_improvement_rate(df, output_path)
        
        # Dashboard completo
        output_path = os.path.join(args.output_dir, 'dashboard.png') if args.output_dir else None
        plot_all_metrics_dashboard(df, output_path)
    
    print("\n✓ Visualização concluída!")


if __name__ == '__main__':
    main()
