#!/usr/bin/env python3
"""
Generate Typst tables comparing query execution time from optimization results JSON.

Usage:
    python generate_typst_time_tables.py results/ranx/optimizing_ranx_group3.json
    python generate_typst_time_tables.py results/ranx/optimizing_ranx_group3.json --output time_tables.typ
"""

import json
import sys
from pathlib import Path
from typing import Any


def format_time_value(value: float | None, decimals: int = 3, is_best_category: bool = False, is_best_global: bool = False) -> str:
    """Format a time metric value for display.

    Args:
        value: The time value in milliseconds
        decimals: Number of decimal places
        is_best_category: True if this is the best (fastest) value within its category
        is_best_global: True if this is the best (fastest) value overall across all categories
    """
    if value is None:
        return ""
    formatted = f"{value:.{decimals}f}"

    if is_best_global:
        # Best overall - use green rectangle
        return f"#rect(fill: rgb(\"#90EE90\"), inset: 1pt)[{formatted}]"
    elif is_best_category:
        # Best in category - use bold italic
        return f"*{formatted}*"
    else:
        return formatted


def get_model_display_name(model_data: dict[str, Any], model_type: str) -> str:
    """Get display name for a model."""
    if model_type == "bm25":
        variant = model_data.get("variant", "").upper()

        # Map to descriptive names
        name_map = {
            "ROBERTSON": "BM25 Robertson (Baseline)",
            "ATIRE": "ATIRE (normalização alternativa)",
            "BM25L": "BM25L (length normalization)",
            "BM25+": "BM25+ (_offset_ positivo)",
            "LUCENE": "Lucene (suavização do IDF)",
            "PYSERINI": "Pyserini BM25 com RM3",
            "BMX": "BMX (entropia + semântica)"
        }

        return name_map.get(variant, variant)

    elif model_type == "embedding":
        variant = model_data.get("variant", "")
        model_name = model_data.get("model_name", "")

        # Map to descriptive names
        if "jina" in variant:
            return "jina-embeddings-v3"
        elif "qwen" in variant.lower() and "lamdec" not in variant:
            return "Qwen-Embedding-0.6B"
        elif "alibaba" in variant:
            return "gte-multilingual-base (Alibaba)"
        elif "lamdec-gte" in variant or ("lamdec" in variant and "gte" in model_name):
            return "gte-lamdec-pairs"
        elif "lamdec-gemma" in variant or ("lamdec" in variant and "gemma" in model_name):
            return "gemma-lamdec-pairs"
        elif "lamdec-qwen" in variant or ("lamdec" in variant and "qwen" in model_name.lower()):
            return "qwen-lamdec-pairs"
        else:
            return variant

    elif model_type == "fusion":
        method = model_data.get("fusion_method", "")
        norm = model_data.get("fusion_norm", "")

        # Map to descriptive names
        method_names = {
            "mixed": "_Mixed_ (WMNZ + WSUM)",
            "rrf": "RRF (Reciprocal Rank Fusion)",
            "wsum": "Weighted Sum",
            "wmnz": "WMNZ (Weighted MNZ)",
            "mnz": "MNZ (Min-Non-Zero)",
            "sum": "Sum Fusion",
            "max": "Max Fusion",
            "min": "Min Fusion",
            "med": "Median Fusion",
            "anz": "ANZ (All-Non-Zero)",
            "gmnz": "GMNZ (Geometric MNZ)",
            "isr": "ISR (Inverse Square Rank)",
            "log_isr": "Log ISR",
            "logn_isr": "LogN ISR",
            "bordafuse": "BordaFuse",
            "w_bordafuse": "Weighted BordaFuse",
            "condorcet": "Condorcet",
            "w_condorcet": "Weighted Condorcet",
            "bayesfuse": "BayesFuse",
            "mapfuse": "MAPFuse",
            "posfuse": "PosFuse",
            "probfuse": "ProbFuse",
            "segfuse": "SegFuse",
            "slidefuse": "SlideFuse",
            "rbc": "RBC (Rank-Biased Centroids)"
        }

        return method_names.get(method, method.upper())

    return "Unknown"


def extract_time_metrics(model_data: dict[str, Any]) -> dict[str, float | None]:
    """Extract time metrics from model data."""
    metrics_dict = model_data.get("metrics", {})

    return {
        "mean": metrics_dict.get("query_time_mean_ms", None),
        "median": metrics_dict.get("query_time_median_ms", None),
        "p95": metrics_dict.get("query_time_p95_ms", None),
        "p99": metrics_dict.get("query_time_p99_ms", None),
        "max": metrics_dict.get("query_time_max_ms", None),
    }


def generate_typst_time_table(data: dict[str, Any]) -> str:
    """Generate a Typst table comparing query execution times."""

    # Time metrics to display
    time_metrics = ["mean", "median", "p95", "p99", "max"]
    time_headers = ["*Média*", "*Mediana*", "*P95*", "*P99*", "*Máx*"]

    # Calculate total columns: Type, Model, MAP, + 5 time metrics
    total_cols = 2 + 1 + len(time_metrics)

    lines = [
        "#align(center)[",
        "#text(size: 8.8pt)[",
        "#table(",
        f"      columns: {total_cols},",
        "      column-gutter: 4pt,",
        "      row-gutter: 1pt,",
        "      align: (left, left, center, center, center, center, center, center),",
        "      stroke: none,",
        "      inset: (x, y) => (",
        "        x: if y == 0 or y == 1 or x == 1 { 3pt } else { 0.5pt },",
        "        y: if y == 0 or y == 1 { 4pt } else if y == 2 { 3pt } else if y == 3 { 2.9pt } else if x == 1 { 1.4pt } else { 1.1pt }",
        "      ),",
        "",
        "      // Header rows",
        "      table.hline(stroke: .6pt),",
        "      table.cell(rowspan: 2, [*Tipo*], align: horizon),",
        "      table.cell(rowspan: 2, [*Modelo*], align: horizon),",
        "      table.cell(rowspan: 2, [*MAP*], align: horizon),",
        f"      table.cell(colspan: {len(time_metrics)}, [*Tempo de Consulta (ms)*]),",
        "",
        f"      table.hline(start: 3, end: {3 + len(time_metrics)}, stroke: 0.3pt),",
        "",
        "      // Sub-headers",
    ]

    # Add sub-headers
    lines.append("      " + ", ".join(time_headers) + ",")
    lines.append("")
    lines.append("      table.hline(stroke: .6pt),")
    lines.append(f"      table.cell(colspan: {total_cols}, []),")
    lines.append("")

    # Extract all BM25 results
    all_bm25_results = data.get("all_bm25_results", [])

    # Get unique BM25 variants (keep best for each variant)
    bm25_by_variant = {}
    for result in all_bm25_results:
        variant = result.get("variant")
        if variant not in bm25_by_variant:
            bm25_by_variant[variant] = result
        elif result.get("primary_score", 0) > bm25_by_variant[variant].get("primary_score", 0):
            bm25_by_variant[variant] = result

    # Sort BM25 by MAP descending
    bm25_results = sorted(bm25_by_variant.values(), key=lambda x: x.get("primary_score", 0), reverse=True)

    # Extract all embedding results
    all_embedding_results = data.get("all_embeddings_results", [])
    # Filter out errors
    embedding_results = [e for e in all_embedding_results if e.get("metrics") is not None]
    # Sort by MAP descending
    embedding_results = sorted(embedding_results, key=lambda x: x.get("primary_score", 0), reverse=True)

    # Fusion - get top 5 unique methods
    all_fusion_results = data.get("all_fusion_results", [])

    # Group by fusion_method and keep only the best for each
    fusion_by_method = {}
    for result in all_fusion_results:
        method = result.get("fusion_method")
        if method not in fusion_by_method:
            fusion_by_method[method] = result
        elif result.get("primary_score", 0) > fusion_by_method[method].get("primary_score", 0):
            fusion_by_method[method] = result

    # Sort by MAP and get top 5
    top_fusions = sorted(fusion_by_method.values(), key=lambda x: x.get("primary_score", 0), reverse=True)[:5]

    # First pass: calculate global minimums (fastest) across all categories
    # For time metrics, lower is better
    global_min = {}
    for metric in time_metrics:
        global_min[metric] = float('inf')

    # Check all BM25 results for global min
    for bm25 in bm25_results:
        times = extract_time_metrics(bm25)
        for metric in time_metrics:
            if times[metric] is not None and times[metric] < global_min[metric]:
                global_min[metric] = times[metric]

    # Check all embedding results for global min
    for embedding in embedding_results:
        times = extract_time_metrics(embedding)
        for metric in time_metrics:
            if times[metric] is not None and times[metric] < global_min[metric]:
                global_min[metric] = times[metric]

    # Check all fusion results for global min
    for fusion in top_fusions:
        times = extract_time_metrics(fusion)
        for metric in time_metrics:
            if times[metric] is not None and times[metric] < global_min[metric]:
                global_min[metric] = times[metric]

    # Calculate min values for BM25 category
    bm25_min = {}
    for metric in time_metrics:
        bm25_min[metric] = float('inf')

    for bm25 in bm25_results:
        times = extract_time_metrics(bm25)
        for metric in time_metrics:
            if times[metric] is not None and times[metric] < bm25_min[metric]:
                bm25_min[metric] = times[metric]

    # BM25 models
    lines.append("      // Modelos Esparsos")
    lines.append(f"      table.cell(rowspan: {len(bm25_results)}, [Esparso], align: horizon),")

    for bm25 in bm25_results:
        model_name = get_model_display_name(bm25, "bm25")
        times = extract_time_metrics(bm25)
        map_score = bm25.get("metrics", {}).get("map", None)

        map_val = f"{map_score:.3f}" if map_score is not None else ""

        values = [
            format_time_value(
                times[metric],
                decimals=3,
                is_best_category=(times[metric] == bm25_min[metric]),
                is_best_global=(times[metric] == global_min[metric])
            )
            for metric in time_metrics
        ]

        lines.append(f"      [{model_name}], [{map_val}], " + ", ".join(f"[{v}]" for v in values) + ",")

    lines.append("")
    lines.append(f"      table.cell(colspan: {total_cols}, []),")
    lines.append("      table.hline(stroke: .6pt),")
    lines.append(f"      table.cell(colspan: {total_cols}, []),")
    lines.append("")

    # Calculate min values for Embedding category
    embedding_min = {}
    for metric in time_metrics:
        embedding_min[metric] = float('inf')

    for embedding in embedding_results:
        times = extract_time_metrics(embedding)
        for metric in time_metrics:
            if times[metric] is not None and times[metric] < embedding_min[metric]:
                embedding_min[metric] = times[metric]

    # Embedding models
    lines.append("      // Modelos Semânticos")
    lines.append(f"      table.cell(rowspan: {len(embedding_results)}, [Semântico], align: horizon),")

    for embedding in embedding_results:
        model_name = get_model_display_name(embedding, "embedding")
        times = extract_time_metrics(embedding)
        map_score = embedding.get("metrics", {}).get("map", None)

        map_val = f"{map_score:.3f}" if map_score is not None else ""

        values = [
            format_time_value(
                times[metric],
                decimals=3,
                is_best_category=(times[metric] == embedding_min[metric]),
                is_best_global=(times[metric] == global_min[metric])
            )
            for metric in time_metrics
        ]

        lines.append(f"      [{model_name}], [{map_val}], " + ", ".join(f"[{v}]" for v in values) + ",")

    lines.append("")
    lines.append(f"      table.cell(colspan: {total_cols}, []),")
    lines.append("      table.hline(stroke: .6pt),")
    lines.append(f"      table.cell(colspan: {total_cols}, []),")
    lines.append("")

    # Calculate min values for Fusion category
    fusion_min = {}
    for metric in time_metrics:
        fusion_min[metric] = float('inf')

    for fusion in top_fusions:
        times = extract_time_metrics(fusion)
        for metric in time_metrics:
            if times[metric] is not None and times[metric] < fusion_min[metric]:
                fusion_min[metric] = times[metric]

    # Fusion models
    lines.append("      // Fusão")
    lines.append(f"      table.cell(rowspan: {len(top_fusions) + 1}, [Fusão], align: horizon),")

    for fusion in top_fusions:
        model_name = get_model_display_name(fusion, "fusion")
        times = extract_time_metrics(fusion)
        map_score = fusion.get("metrics", {}).get("map", None)

        map_val = f"{map_score:.3f}" if map_score is not None else ""

        values = [
            format_time_value(
                times[metric],
                decimals=3,
                is_best_category=(times[metric] == fusion_min[metric]),
                is_best_global=(times[metric] == global_min[metric])
            )
            for metric in time_metrics
        ]

        lines.append(f"      [{model_name}], [{map_val}], " + ", ".join(f"[{v}]" for v in values) + ",")

    # "E muitos outros algoritmos..." row
    lines.append("      [E muitos outros algoritmos...], " + ", ".join(["[]"] * (1 + len(time_metrics))) + ",")

    lines.append("")
    lines.append("      table.hline(stroke: .6pt),")
    lines.append("    )")
    lines.append("  ]")
    lines.append("  #v(.4em)")

    # Caption
    query_group = data.get("query_group")
    if isinstance(query_group, list):
        group_str = ", ".join(str(g) for g in query_group)
    else:
        group_str = str(query_group)

    lines.append(f"  #align(center)[Tabela. Tempo de execução de consultas para o grupo {group_str}. Valores em *itálico* indicam o menor tempo (mais rápido) dentro de cada categoria (Esparso, Semântico, Fusão). Valores em #rect(fill: rgb(\"#90EE90\"), inset: 1pt)[verde] indicam o menor tempo global.]")
    lines.append("]")
    lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_typst_time_tables.py <json_file> [--output <output_file>]")
        print("Example: python generate_typst_time_tables.py results/ranx/optimizing_ranx_group3.json")
        sys.exit(1)

    json_file = sys.argv[1]

    # Check for output file
    output_file = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]

    # Read JSON
    with open(json_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded data from {json_file}")
    print(f"Query Group: {data.get('query_group')}")
    print(f"Evaluation Library: {data.get('evaluation_library', 'unknown')}")
    print()

    # Generate table
    print("Generating time comparison table...")
    time_table = generate_typst_time_table(data)

    # Output
    if output_file:
        with open(output_file, 'w') as f:
            f.write(time_table)
        print(f"\nTime table written to {output_file}")
    else:
        print("\n" + "="*80)
        print("TYPST TIME TABLE OUTPUT")
        print("="*80 + "\n")
        print(time_table)
        print("\n" + "="*80)

        # Auto-save to default location
        default_output = Path(json_file).parent / f"{Path(json_file).stem}_time_table.typ"
        with open(default_output, 'w') as f:
            f.write(time_table)
        print(f"\nTime table also saved to {default_output}")


if __name__ == "__main__":
    main()
