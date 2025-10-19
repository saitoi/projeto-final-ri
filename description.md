# Análise da Eficácia de Interpolação com Scores do BM25 no Contexto Jurídico Brasileiro

Busco avaliar a efetividade de técnicas de interpolação dos scores do BM25 com
modelos de embeddings (GTE, JINA e GTE fine-tuned) no domínio jurídico
brasileiro. A avaliação será feita sobre dois datasets complementares que
representam diferentes cenários do contexto jurídico brasileiro.

## Modelos

### 1. Alibaba-NLP/gte-multilingual-base

- **Arquitetura**: Adaptação do BERT
- **Parâmetros**: 305 milhões
- **Janela de contexto**: 8192 tokens
- **Dimensão de representação**: 768 tokens (adaptável)
- **Características**: Modelo multilingual com boa performance em português

### 2. jinaai/jina-embeddings-v3

- **Arquitetura**: Baseado em XLM-RoBERTa
- **Parâmetros**: 572 milhões
- **Janela de contexto**: 8192 tokens
- **Dimensão de representação**: 1024 tokens (adaptável)
- **Características**: Inclui adaptadores LoRA para diferentes tarefas

### 3. gte-finetune-pgm (Modelo ajustado)

- **Base**: gte-multilingual-base
- **Fine-tuning**: Especializado para domínio jurídico da PGM-Rio
- **Técnicas utilizadas**:
  - Multiple Negatives Ranking Loss (MNRL)
  - Matryoshka Representation Learning (MRL)
  - Hard negatives baseados em metadados
- **Dimensões disponíveis**: [768, 512, 256, 128, 64]

## Datasets

### 1. JurisTCU
Dataset em português brasileiro para recuperação da informação jurídica do Tribunal de Contas da União.

**Características:**
- **Documentos**: 16.045 documentos de jurisprudência selecionada
- **Queries**: 150 consultas com julgamentos de relevância
- **Grupos de queries**:
  - **G1**: 50 queries reais de usuários (formato keyword, média de 3.5 palavras)
  - **G2**: 50 queries sintéticas keyword-based (média de 6.5 palavras)
  - **G3**: 50 queries sintéticas question-based (média de 16.5 palavras)
- **Anotação**: Escala de 0-3 (irrelevante → altamente relevante)
- **Método de anotação**: Híbrido (LLM + validação manual por especialista)

**Estrutura dos documentos:**
- Enunciado (SUMMARY): resumo da decisão (~47 palavras)
- Excerto (EXCERPT): texto original que fundamenta o entendimento (~660 palavras)
- Metadados: área, tema, subtema, relator, tipo de processo, etc.

### 2. Dados PGM-Rio
Acervo da Procuradoria Geral do Município do Rio de Janeiro.

**Características:**
- **Documentos**: 261.325 documentos (após filtragem)
- **Processos**: 26.286 processos diferentes
- **Segmentos**: 2.145.959 segmentos de 1024 tokens
- **Formato**: Documentos digitais (PDF e HTML) ocerizados e/ou extraídos.
- **Metadados principais**:
  - `especializada`: Procuradoria responsável
  - `tipo`: Tipo do documento (intimação, acórdão, decisão)
  - `assunto_principal`: Assunto do processo
  - `tipo_processo`: Tipo do processo

**Tratamento dos dados:**

- Foco exclusivo em documentos digitais (sem OCR) [Não estamos fazendo]
- Segmentação recursiva respeitando limites semânticos
- Agregação via mean pooling para representação de documentos completos

## Metodologia

### Abordagens de Busca

**1. Busca interpolada entre BM25 e Dense Retrievers**

...

<!--**1. Busca Lexical (BM25 com Expansão de Documentos)**
- BM25 baseline (k1=1.2, b=0.75)
- Document expansion via docT5query
- Document expansion via sinônimos (GPT-3.5, GPT-4o, Llama 3-70B)
- Combinação de ambas as técnicas-->

**2. Busca Semântica (Dense Retrievers)**

- Embeddings BERT-based em português
- Embeddings OpenAI (text-embedding-3-small e text-embedding-3-large)
- Fine-tuned model (gte-finetune-pgm) em múltiplas dimensões

### Estratégia de Interpolação

$$
s(p) = \alpha \cdot \text{ŝBM25}(p) + (1 - \alpha) \cdot \text{sDR}(p)
$$

onde:
- $\alpha$ varia de 0 a 1 (passo de 0.1)
- $ŝBM25(p)$: score normalizado do BM25
- $sDR(p)$: score do dense retriever

## Medidas de Avaliação

### Medidas Rasas (Shallow Metrics)

- **Precision@10**: Precisão nos top-10 resultados
- **MRR@10** (Mean Reciprocal Rank): Posição do primeiro documento relevante
- **nDCG@10** (Normalized Discounted Cumulative Gain): Qualidade do ranking considerando ordem e relevância

### Medidas Profundas (Deep Metrics)

- **Precision@20, Precision@1000**
- **Recall@1000**: Cobertura dos documentos relevantes
- **MAP** (Mean Average Precision): Precisão média em todos os níveis de recall
- **nDCG@1000**: Qualidade do ranking em profundidade

## Relação com o Artigo Base

O artigo base **"BERT-based Dense Retrievers Require Interpolation with BM25 for Effective Passage Retrieval"** (Wang et al., ICTIR 2021) investiga se dense retrievers baseados em BERT (RepBERT e ANCE) capturam os mesmos sinais de relevância que o BM25, e se a interpolação entre os scores melhora a eficácia da recuperação.

### Principais Descobertas do Artigo Base:
1. **Dense retrievers necessitam interpolação com BM25**, diferentemente do BERT re-ranker
2. **DRs são excelentes para sinais fortes de relevância**, mas falham em sinais fracos
3. **Interpolação significativa**: Ganhos de até 48.5% em MAP (TREC DL 2019)
4. **Dimensão profunda importa**: Melhoria substancial em métricas profundas (MAP, nDCG@1000)

### Diferenças e Contribuições do Projeto:
- **Contexto**: Aplicação ao domínio jurídico brasileiro (vs. MS MARCO em inglês)
- **Datasets especializados**: JurisTCU e PGM-Rio (vs. datasets genéricos)
- **Modelos multilíngues**: Foco em modelos com suporte a português
- **Fine-tuning de domínio**: Avaliação de modelo especializado (gte-finetune-pgm)
- **Múltiplas dimensões**: Análise do trade-off qualidade vs. custo computacional via MRL
- **Tipos variados de queries**: Comparação entre queries reais, sintéticas keyword e question-based

## Resultados Esperados

Com base nos estudos preliminares dos artigos:

- Confirmação da necessidade de interpolação no contexto jurídico brasileiro
- **Ganhos significativos com fine-tuning**: Esperado ~23-28% de melhoria em nDCG@20
- **Eficácia do MRL**: Embeddings de menor dimensão (256d) com performance próxima à dimensão máxima (768d)
<!--4. **Superioridade de modelos OpenAI**: Especialmente em queries curtas (~70% de melhoria)-->
<!--5. **Document expansion eficaz**: Melhorias de 45%+ em queries keyword curtas com docT5query + sinônimos-->

## Questões de Pesquisa

Questões de pesquisa inspiradas nas do [artigo base](./articles/article-base.pdf):

1. **RQ1**: Dense retrievers especializados em português capturam os mesmos sinais de relevância que o BM25 no contexto jurídico?

2. **RQ2**: Os ganhos observados em medidas rasas se generalizam para medidas profundas?

3. **RQ3**: O fine-tuning de domínio melhora a capacidade de interpolação dos dense retrievers?

4. **RQ4**: Qual o impacto do tipo de query (real vs. sintética, keyword vs. question) na eficácia da interpolação?

5. **RQ5**: Qual o trade-off ótimo entre dimensionalidade e qualidade via Matryoshka embeddings?

**Obs**: As questões de pesquisa levantadas estão sujeitas a mudanças.

## Referências

- [Artigo base](./articles/article-base.pdf)
- [Artigo sobre dataset do TCU](./articles/article-tcu.pdf)
- [Artigo sobre fine-tuning de DRs no contexto jurídico](./articles/article-matheus.pdf)
