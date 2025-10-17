# Pitch de 2 Minutos

## Interpolação BM25 + Dense Retrievers no Contexto Jurídico Brasileiro

### O Problema

Modelos de recuperação densa baseados em BERT revolucionaram a recuperação de informação,
mas pesquisas recentes (Wang et al., 2021) mostram que eles **necessitam
interpolação com BM25** para serem eficazes. Enquanto capturam bem sinais
fortes de relevância semântica, falham em sinais fracos. Será que isso se confirma no
domínio jurídico brasileiro, especialmente com modelos especializados via
fine-tuning?

### Proposta

Avaliar a eficácia da interpolação entre BM25 e modelos de embeddings (GTE,
JINA e GTE fine-tuned) em **português**, usando dois datasets jurídicos
brasileiros:

**1. JurisTCU** - 16k documentos do TCU, 150 queries (reais + sintéticas)
**2. PGM-Rio** - 261k documentos da Procuradoria do Rio

### Metodologia

Comparamos três abordagens:
- **Interpolação entre BM25 e DRs**.
- **Dense retrievers** (BERT-based e OpenAI embeddings)
- **Modelo fine-tuned** (gte-finetune-pgm) usando MNRL + Matryoshka Learning

Avaliamos com métricas rasas (P@10, MRR@10, nDCG@10) e profundas (MAP,
Recall@1000, nDCG@1000), variando o parâmetro de interpolação $\alpha$ de 0 a 1.

### Resultados Esperados

Baseado em experimentos preliminares:
- **23-28%** de ganho em nDCG@20 com fine-tuning de domínio
- **70%** de melhoria com embeddings OpenAI em queries curtas
- **45%+** de ganho com document expansion em queries keyword
- Embeddings de **256 dimensões** mantêm ~95% da performance de 768d

### Contribuições

1. Primeiro estudo de interpolação BM25+DRs para **português jurídico**
2. Avaliação de **fine-tuning de domínio** na capacidade de interpolação
3. Análise do trade-off **qualidade vs. custo** via Matryoshka embeddings
4. Benchmark com **tipos variados de queries** (reais, sintéticas keyword, question-based)
