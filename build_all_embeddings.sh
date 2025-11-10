#!/bin/bash

# Script para gerar todos os embeddings
# Em caso de erro, pula para o próximo modelo

# Lista de variantes de embeddings
VARIANTS=(
    "jina"
    "alibaba"
    "lamdec"
    "gemma"
    "qwen"
    "lamdec-qwen"
    "lamdec-gemma"
    "lamdec-gte"
)

DATABASE="dataset.duckdb"
LOG_DIR="logs/embeddings"
mkdir -p "$LOG_DIR"

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Building All Embedding Variants"
echo "=========================================="
echo "Total variants: ${#VARIANTS[@]}"
echo "Database: $DATABASE"
echo "Logs directory: $LOG_DIR"
echo ""

SUCCESSFUL=0
FAILED=0
FAILED_VARIANTS=()

for variant in "${VARIANTS[@]}"; do
    echo ""
    echo "=========================================="
    echo -e "${YELLOW}Processing variant: $variant${NC}"
    echo "=========================================="

    LOG_FILE="$LOG_DIR/${variant}_$(date +%Y%m%d_%H%M%S).log"

    # Executa o comando e captura o status
    if uv run src/search/generate_embeddings.py \
        --database "$DATABASE" \
        --embedding_variant "$variant" \
        --build true \
        2>&1 | tee "$LOG_FILE"; then

        echo -e "${GREEN}✓ Successfully built embeddings for: $variant${NC}"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo -e "${RED}✗ Failed to build embeddings for: $variant${NC}"
        echo -e "${RED}  Check log: $LOG_FILE${NC}"
        FAILED=$((FAILED + 1))
        FAILED_VARIANTS+=("$variant")
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "${GREEN}Successful: $SUCCESSFUL${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed variants:"
    for variant in "${FAILED_VARIANTS[@]}"; do
        echo -e "  ${RED}- $variant${NC}"
    done
fi

echo ""
echo "All logs saved to: $LOG_DIR"
echo "=========================================="

exit 0
