.PHONY: help install test run clean

# Python
VENV_PYTHON := .venv/bin/python
PIP := .venv/bin/pip

# Diretórios
EXPERIMENTS_DIR := experiments

# ============================================================================
# COMANDOS PRINCIPAIS
# ============================================================================

help: ## Mostra esta mensagem de ajuda
	@echo "================================================================"
	@echo "  Algoritmo Genético para Alinhamento Múltiplo de Sequências"
	@echo "================================================================"
	@echo ""
	@echo "Comandos:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'
	@echo ""

install: ## Instala dependências
	@echo "Instalando dependências..."
	@test -d .venv || python3 -m venv .venv
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "✓ Instalado!"

test: ## Executa testes
	@$(VENV_PYTHON) -m pytest tests/ -v

run: ## Executa experimentos (use: make run SEQ=10 RUNS=3)
	@$(VENV_PYTHON) run_all_experiments.py \
		--max-sequences $(or $(SEQ),10) \
		--num-runs $(or $(RUNS),3) \
		--output-dir $(EXPERIMENTS_DIR)

analyze: ## Analisa último experimento
	@LAST=$$(ls -t $(EXPERIMENTS_DIR) 2>/dev/null | head -1); \
	if [ -n "$$LAST" ] && [ -f "$(EXPERIMENTS_DIR)/$$LAST/SUMMARY_REPORT.md" ]; then \
		cat "$(EXPERIMENTS_DIR)/$$LAST/SUMMARY_REPORT.md"; \
	else \
		echo "Nenhum experimento encontrado. Execute: make run"; \
	fi

visualize: ## Gera gráficos do último experimento
	@LAST=$$(ls -t $(EXPERIMENTS_DIR) 2>/dev/null | head -1); \
	if [ -n "$$LAST" ]; then \
		mkdir -p visualizations/$$LAST; \
		for run_dir in $(EXPERIMENTS_DIR)/$$LAST/*/run_1/; do \
			if [ -f "$$run_dir/generation_history.csv" ]; then \
				exp=$$(basename $$(dirname $$run_dir)); \
				$(VENV_PYTHON) visualize_metrics.py "$$run_dir/generation_history.csv" \
					--output-dir "visualizations/$$LAST/$$exp"; \
			fi; \
		done; \
		echo "✓ Gráficos em: visualizations/$$LAST/"; \
	else \
		echo "Nenhum experimento encontrado."; \
	fi

clean: ## Remove resultados
	@read -p "Remover todos os experimentos? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(EXPERIMENTS_DIR) visualizations; \
		find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true; \
		echo "✓ Limpo!"; \
	fi
