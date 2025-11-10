#!/bin/bash

# Script completo para avaliar todos os modelos em todos os grupos
# - Grid search BM25 para cada grupo (0, 1, 2)
# - Benchmark embeddings para cada grupo
# - Benchmark hybrid (RRF) para todas as combinações BM25 x Embeddings

set -e  # Exit on error (mas vamos capturar erros manualmente)

# Configurações
DATABASE="dataset.duckdb"
RESULTS_DIR="results"
LOGS_DIR="logs/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Criar diretórios
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

# Cores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Listas de modelos
BM25_VARIANTS=("robertson" "lucene" "atire" "bm25l" "bm25+" "pyserini")
EMBEDDING_VARIANTS=("jina" "alibaba" "lamdec" "gemma" "qwen" "lamdec-qwen" "lamdec-gemma" "lamdec-gte")
QUERY_GROUPS=(1 2 3)

# Contadores
TOTAL_TASKS=0
COMPLETED_TASKS=0
FAILED_TASKS=0

# Log principal
MAIN_LOG="$LOGS_DIR/full_benchmark_${TIMESTAMP}.log"

log() {
    echo -e "$1" | tee -a "$MAIN_LOG"
}

log_header() {
    log ""
    log "=========================================="
    log "$1"
    log "=========================================="
}

run_task() {
    local task_name="$1"
    local task_cmd="$2"
    local log_file="$3"

    TOTAL_TASKS=$((TOTAL_TASKS + 1))

    log ""
    log "${YELLOW}[$TOTAL_TASKS] Running: $task_name${NC}"
    log "Command: $task_cmd"
    log "Log: $log_file"

    if eval "$task_cmd" > "$log_file" 2>&1; then
        COMPLETED_TASKS=$((COMPLETED_TASKS + 1))
        log "${GREEN}✓ Success: $task_name${NC}"
        return 0
    else
        FAILED_TASKS=$((FAILED_TASKS + 1))
        log "${RED}✗ Failed: $task_name${NC}"
        log "${RED}  Check log: $log_file${NC}"
        return 1
    fi
}

# Início
log_header "Full Benchmark Suite - Started at $(date)"
log "Database: $DATABASE"
log "Results: $RESULTS_DIR"
log "Logs: $LOGS_DIR"

# ==========================================
# 1. BM25 Grid Search (todos os grupos)
# ==========================================
log_header "Phase 1: BM25 Grid Search"
log "Variants: ${BM25_VARIANTS[*]}"
log "Query groups: ${QUERY_GROUPS[*]}"
log "Note: Pyserini will be skipped (no custom parameters)"

for group in "${QUERY_GROUPS[@]}"; do
    log ""
    log "${BLUE}>>> Query Group $group <<<${NC}"

    # Grid search para todos os BM25 variants (exceto pyserini)
    task_name="BM25 Grid Search - Group $group"
    task_cmd="BM25_GRID_SEARCH=1 uv run src/benchmarks/main.py --variant bm25 --query_group $group"
    log_file="$LOGS_DIR/bm25_grid_search_group${group}_${TIMESTAMP}.log"

    run_task "$task_name" "$task_cmd" "$log_file"
done

# ==========================================
# 2. Embeddings Benchmark (todos os grupos)
# ==========================================
log_header "Phase 2: Embeddings Benchmark"
log "Variants: ${EMBEDDING_VARIANTS[*]}"
log "Query groups: ${QUERY_GROUPS[*]}"

for variant in "${EMBEDDING_VARIANTS[@]}"; do
    for group in "${QUERY_GROUPS[@]}"; do
        task_name="Embeddings ($variant) - Group $group"
        task_cmd="uv run src/benchmarks/main.py --variant embeddings --embedding_variant $variant --query_group $group -k 1000"
        log_file="$LOGS_DIR/embeddings_${variant}_group${group}_${TIMESTAMP}.log"

        run_task "$task_name" "$task_cmd" "$log_file" || continue
    done
done

# ==========================================
# 3. Hybrid RRF Benchmark (todas as combinações)
# ==========================================
log_header "Phase 3: Hybrid RRF Benchmark"
log "BM25 Variants: ${BM25_VARIANTS[*]}"
log "Embedding Variants: ${EMBEDDING_VARIANTS[*]}"
log "Query groups: ${QUERY_GROUPS[*]}"
log "Total combinations per group: $((${#BM25_VARIANTS[@]} * ${#EMBEDDING_VARIANTS[@]}))"

for group in "${QUERY_GROUPS[@]}"; do
    log ""
    log "${BLUE}>>> Query Group $group - Hybrid RRF Combinations <<<${NC}"

    for bm25_var in "${BM25_VARIANTS[@]}"; do
        for emb_var in "${EMBEDDING_VARIANTS[@]}"; do
            task_name="Hybrid RRF ($bm25_var + $emb_var) - Group $group"
            task_cmd="uv run src/benchmarks/main.py --variant hybrid --hybrid_variant rrf --bm25_variant $bm25_var --embedding_variant $emb_var --query_group $group -k 1000"
            log_file="$LOGS_DIR/hybrid_rrf_${bm25_var}_${emb_var}_group${group}_${TIMESTAMP}.log"

            run_task "$task_name" "$task_cmd" "$log_file" || continue
        done
    done
done

# ==========================================
# Summary
# ==========================================
log_header "Benchmark Suite Complete!"
log "Completed at: $(date)"
log ""
log "${GREEN}Completed tasks: $COMPLETED_TASKS${NC}"
log "${RED}Failed tasks: $FAILED_TASKS${NC}"
log "Total tasks: $TOTAL_TASKS"
log ""
log "Results saved to: $RESULTS_DIR"
log "Logs saved to: $LOGS_DIR"
log "Main log: $MAIN_LOG"

# Generate summary report
SUMMARY_FILE="$RESULTS_DIR/benchmark_summary_${TIMESTAMP}.txt"
{
    echo "=========================================="
    echo "Full Benchmark Summary"
    echo "=========================================="
    echo "Date: $(date)"
    echo ""
    echo "Completed: $COMPLETED_TASKS"
    echo "Failed: $FAILED_TASKS"
    echo "Total: $TOTAL_TASKS"
    echo ""
    echo "=========================================="
    echo "Results Files:"
    echo "=========================================="
    find "$RESULTS_DIR" -type f -name "*.json" -newer "$MAIN_LOG" | sort
    echo ""
    echo "=========================================="
    echo "Check main log for details:"
    echo "$MAIN_LOG"
    echo "=========================================="
} > "$SUMMARY_FILE"

log ""
log "Summary report: $SUMMARY_FILE"
log ""

if [ $FAILED_TASKS -gt 0 ]; then
    log "${YELLOW}Warning: Some tasks failed. Check logs for details.${NC}"
    exit 1
else
    log "${GREEN}All tasks completed successfully!${NC}"
    exit 0
fi
