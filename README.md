# Algoritmo Genético para Alinhamento Múltiplo de Sequências

Implementação de um Algoritmo Genético para resolver o problema de Alinhamento Múltiplo de Sequências (MSA), comparando diferentes estratégias de operadores genéticos e funções objetivo através de experimentos controlados.

## 🚀 Início Rápido

```bash
make install        # Instalar dependências
make run SEQ=5      # Executar experimentos (5 sequências, teste rápido)
make analyze        # Ver relatório dos resultados
make visualize      # Gerar gráficos comparativos
```

---

## 🔬 Metodologia Experimental

O sistema executa **experimentos comparativos sistemáticos** para avaliar diferentes componentes do algoritmo genético. Cada configuração é executada múltiplas vezes para garantir significância estatística.

### Configuração Base

Todos os experimentos utilizam os seguintes parâmetros padrão (quando não variados):

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `population_size` | 30 | Tamanho da população |
| `num_generations` | 50 | Número de gerações |
| `crossover_probability` | 0.8 | Probabilidade de cruzamento |
| `mutation_probability` | 0.5 | Probabilidade de mutação |
| `elitism_count` | 2 | Número de elites preservados |
| `num_runs` | 3 | Repetições por configuração |

---

## 📊 Categorias de Experimentos

### 1. Funções Objetivo (SAGA vs T-Coffee)

Compara duas abordagens filosóficas diferentes para avaliar a qualidade de alinhamentos.

#### SAGA (Notredame & Higgins, 1996)
- **Base**: Matriz de substituição PAM250
- **Abordagem**: Conhecimento biológico sobre evolução molecular
- **Cálculo**: Soma de scores de pares de aminoácidos + penalidades de gaps
- **Penalidades**: Gap opening (-11) + gap extension (-1)

#### T-Coffee (Notredame et al., 2000)
- **Base**: Biblioteca de alinhamentos par-a-par
- **Abordagem**: Consistência entre múltiplos alinhamentos
- **Cálculo**: Média dos scores de consistência par-a-par
- **Método**: Constrói biblioteca com alinhamentos locais (Biopython)

**Configurações testadas**: `obj_saga`, `obj_tcoffee`

---

### 2. Operadores de Mutação

Avalia diferentes estratégias de mutação, desde básicas até guiadas biologicamente.

#### Operadores Básicos

- **`standard`**: Mutação clássica (troca posições + mutação pontual)
- **`gap_shift`**: Desloca gaps uma posição à esquerda/direita
- **`insertion_deletion`**: Adiciona/remove colunas inteiras de gaps

#### Operadores SAGA (Biologicamente Guiados)

- **`saga_gap_insertion`**: Usa PAM250 e árvore filogenética para inserir gaps em regiões menos conservadas
- **`saga_block_shuffling`**: Embaralha blocos de colunas conservadas (16 variantes)
- **`saga_mixed`**: Combina dinamicamente todos os operadores SAGA

**Configurações testadas**: `mut_standard`, `mut_gap_shift`, `mut_insertion_deletion`, `mut_saga_gap`, `mut_saga_block`, `mut_saga_mixed`

**Por que separados?** Cada experimento usa UM operador para isolar seu efeito e comparar qual é melhor.

---

### 3. Métodos de Seleção

Compara estratégias de pressão seletiva na evolução da população.

- **`tournament`**: Seleciona o melhor de 3 indivíduos aleatórios (pressão moderada)
- **`roulette`**: Probabilidade proporcional ao fitness (pode convergir prematuramente)
- **`rank`**: Baseado no ranking, não no valor absoluto (pressão constante)

**Configurações testadas**: `sel_tournament`, `sel_roulette`, `sel_rank`

---

### 4. Métodos de Crossover

Avalia diferentes estratégias de recombinação genética.

- **`default`**: Baseado em blocos de colunas (preserva estrutura local)
- **`single_point`**: Corta em um ponto e combina (preserva grandes segmentos)
- **`uniform`**: Cada coluna vem de um pai (máxima mistura, normaliza tamanhos)

**Configurações testadas**: `cross_default`, `cross_single_point`, `cross_uniform`

---

### 5. Sensibilidade de Parâmetros

Analisa impacto dos hiperparâmetros na qualidade do alinhamento.

#### Tamanho da População
- **20**: Mais rápido, menos diversidade
- **30**: Balanceamento (padrão)
- **50**: Maior exploração, mais lento

#### Probabilidade de Mutação
- **0.3**: Convergência rápida
- **0.5**: Balanceamento (padrão)
- **0.7**: Mais exploração

#### Probabilidade de Crossover
- **0.6**: Menos recombinação
- **0.8**: Padrão recomendado
- **0.9**: Máxima recombinação

**Configurações testadas**: `param_pop_{20,30,50}`, `param_mut_{30,50,70}`, `param_cross_{60,80,90}`

---

## 📈 Métricas Coletadas

Para cada execução:

- **`best_fitness_overall`**: Melhor fitness global (nunca diminui)
- **`best_fitness_generation`**: Melhor fitness da geração atual
- **`avg_fitness`**: Fitness médio da população
- **`diversity`**: Diversidade genética (0=idêntica, 1=diversa)
- **`improvement`**: Melhoria absoluta
- **`improvement_percent`**: Melhoria percentual (métrica principal)
- **`elapsed_time_seconds`**: Tempo de execução

---

## 🎯 Estrutura de Resultados

```
experiments/YYYYMMDD_HHMMSS/
├── consolidated_results.csv         # Tabela com todos os resultados
├── consolidated_results.json        # Resultados detalhados
├── SUMMARY_REPORT.md                # Relatório comparativo
├── visualizations/                  # Gráficos
│   ├── comparison_obj.png
│   ├── comparison_mut.png
│   ├── comparison_sel.png
│   ├── comparison_cross.png
│   └── comparison_param.png
└── [experiment_name]/
    ├── run_1/
    │   └── YYYYMMDD_HHMMSS/
    │       ├── generation_history.csv
    │       ├── final_alignment.txt
    │       ├── best_alignment_info.json
    │       └── metadata.json
    ├── run_2/
    └── run_3/
```

---

## 📊 Análise dos Resultados

### Visualizar Relatório

```bash
make analyze
```

Exibe o `SUMMARY_REPORT.md` com estatísticas e rankings.

### Gerar Gráficos

```bash
make visualize
```

Cria visualizações em `experiments/<ID>/visualizations/`.

### Análise Customizada

```python
import pandas as pd

# Carregar resultados
df = pd.read_csv("experiments/<ID>/consolidated_results.csv")

# Comparar funções objetivo
obj_comp = df[df['experiment_name'].str.startswith('obj_')]
print(obj_comp.groupby('experiment_name')['improvement_percent'].mean())

# Melhor mutação
best_mut = df[df['experiment_name'].str.startswith('mut_')] \
    .groupby('experiment_name')['improvement_percent'] \
    .mean().idxmax()
print(f"Melhor mutação: {best_mut}")
```

---

## 🛠️ Comandos Disponíveis

```bash
make help          # Ver todos os comandos
make install       # Instalar dependências
make test          # Executar testes unitários

# Executar experimentos
make run           # Padrão: 10 sequências, 3 runs
make run SEQ=15    # Customizar número de sequências
make run RUNS=5    # Customizar número de repetições

# Analisar resultados
make analyze       # Ver relatório
make visualize     # Gerar gráficos

# Limpeza
make clean         # Remover todos os resultados
```

### Execução Manual

```bash
source .venv/bin/activate

python run_all_experiments.py \
    --sequences-file sequences/seqdump_1.txt \
    --max-sequences 10 \
    --num-runs 3 \
    --output-dir experiments
```

---

## 📚 Referências

1. **Notredame, C., & Higgins, D. G. (1996).** SAGA: Sequence Alignment by Genetic Algorithm. *Nucleic Acids Research*, 24(8), 1515-1524.

2. **Notredame, C., Higgins, D. G., & Heringa, J. (2000).** T-Coffee: A novel method for fast and accurate multiple sequence alignment. *Journal of Molecular Biology*, 302(1), 205-217.

3. **Dayhoff, M. O., Schwartz, R. M., & Orcutt, B. C. (1978).** A model of evolutionary change in proteins. *Atlas of Protein Sequence and Structure*, 5(3), 345-352.

---

## 📁 Estrutura do Código

```
sequence_alignment/
├── genetic_algorithm/
│   ├── genetic_algorithm.py          # Orquestrador principal
│   ├── alignment.py                  # Classe Alignment
│   ├── objective_function/
│   │   ├── saga_objective_function.py
│   │   ├── tcoffee_objective_function.py
│   │   └── pam250.py
│   ├── operators/
│   │   ├── selection.py
│   │   ├── crossover.py
│   │   ├── mutation.py
│   │   └── saga_operators.py
│   └── utils/
│       ├── read_sequences_file.py
│       └── phylogenetic_tree.py
├── sequences/
│   └── seqdump_1.txt                 # 100 sequências
├── tests/
├── run_all_experiments.py            # Executor
├── visualize_metrics.py              # Gráficos
└── main.py                           # Exemplo
```

---

