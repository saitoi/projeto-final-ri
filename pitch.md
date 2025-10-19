# Pitch

## Interpolação BM25 + Dense Retrievers no Contexto Jurídico Brasileiro

<!--### O Problema-->
<!--Modelos de recuperação densa baseados em BERT revolucionaram a recuperação de informação,-->
<!--mas pesquisas recentes (Wang et al., 2021) mostram que eles **necessitam-->
<!--interpolação com BM25** para serem eficazes. Enquanto capturam bem sinais-->
<!--fortes de relevância semântica, falham em sinais fracos. Será que isso se confirma no-->
<!--domínio jurídico brasileiro, especialmente com modelos especializados via-->
<!--fine-tuning?-->

### Proposta

Avaliar a eficácia da interpolação entre modelos de recuperação densa baseados
em BERT e modelos tradicionais de recuperação da informação (BM25) no contexto
jurídico brasileiro. Modelos de _embeddings_ costumam capturar bem sinais fortes
de relevância semântica, mas tendem a falhar em sinais fracos associados a
termos específicos, uma limitação que o BM25 pode compensar.

A interpolação é realizada da seguinte forma:

$$
score = \alpha \cdot s_{\text{BM25}} (p) + (1 - \alpha) \cdot s_{\text{BERT}} (p)
$$

### Metodologia

O dataset utilizado será o JurisTCU, composto por 16 mil documentos
jurisprudenciais do Tribunl de Contas da União e 150 consultas anotadas com
julgamento de relevância.

As consultas são divididas em três categorias:

- Consultas de usuários.
- Consultas sintéticas de palavras-chave.
- Consultas sintéticas de frases completas.

Entre os modelos de embeddings que serão avaliados, estão:

- General Text Embeddings (GTE) do Alibaba.
- JINA Embeddings (JINA) do Jina AI.
- Versão finetuned do GTE (GTE-Finetuned) treinado com o linguajar jurídico.

As três abordagens avaliadas serão:

- BM25 (_baseline_).
- Interpolação de scores entre BM25 e DRs (baseados ou não em BERT).
- Interpolação de scores entre BM25 e embeddings densos.
- Interpolação de scores entre BM25 e embeddings fine-tuned.

> [!NOTE]
> Pretendo avaliar o desempenho da recuperação variando o parâmetro de
> interpolação $\alpha$ no intervalo de 0 a 1.

As métricas utilizadas incluem:

- Métricas rasas: P@10, MRR@10, nDCG@10.
- Métricas profundas: MAP, Recall@1000, nDCG@1000.

O modelo fine-tuned (gte-finetune-pgm) foi treinado com a função de perda
Multiple Negative Ranking Loss (MNRL) juntamente do Matryoshka Learning que
permite ajustar os embeddings a diferentes dimensões dependendo dos recursos
computacionais disponíveis.

### Relação com o Artigo Base

O artigo base (_Recuperadores densos baseados em BERT exigem interpolação com
BM25 para recuperação efetiva de passagens_) identificou ganhos expressivos ao
interpolar scores entre BM25 e DRs baseados em BERT (RepBERT, ANCE e CLEAR),
sobretudo em métricas profundas.

Embora o presente trabalho se baseie fortemente nesse artigo, também incorpora ideias de outros dois estudods:

1. _JurisTCU: Um conjunto de dados de recuperação de informação em português do Brasil com julgamentos de relevância de consulta_.
2. _Análise da Eficácia de Fine-Tuning de Embeddings no Contexto Jurídico Brasileiro_.

<!--### Resultados Esperados-->
<!--Baseado em experimentos preliminares:-->
<!--- **23-28%** de ganho em nDCG@20 com fine-tuning de domínio-->
<!--- **70%** de melhoria com embeddings OpenAI em queries curtas-->
<!--- **45%+** de ganho com document expansion em queries keyword-->
<!--- Embeddings de **256 dimensões** mantêm ~95% da performance de 768d-->
<!--### Contribuições-->
<!--1. Primeiro estudo de interpolação BM25+DRs para **português jurídico**-->
<!--2. Avaliação de **fine-tuning de domínio** na capacidade de interpolação-->
<!--3. Análise do trade-off **qualidade vs. custo** via Matryoshka embeddings-->
<!--4. Benchmark com **tipos variados de queries** (reais, sintéticas keyword, question-based)-->-->