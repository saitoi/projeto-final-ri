#!/usr/bin/env python3
"""
Generate Typst tables from optimization results JSON.

Usage:
    python generate_typst_tables.py results/ranx/optimizing_ranx_group3.json
    python generate_typst_tables.py results/ranx/optimizing_ranx_group3.json --output tables.typ
"""

import json
import sys
from pathlib import Path
from typing import Any


def format_value(value: float | None, decimals: int = 3, is_best_category: bool = False, is_best_global: bool = False) -> str:
    """Format a metric value for display.

    Args:
        value: The metric value
        decimals: Number of decimal places
        is_best_category: True if this is the best value within its category (Esparso, Semântico, Fusão)
        is_best_global: True if this is the best value overall across all categories
    """
    if value is None:
        return ""
    formatted = f"{value:.{decimals}f}"

    if is_best_global:
        # Best overall - use golden rectangle
        return f"#rect(fill: rgb(\"#DAA520\"), inset: 1pt)[{formatted}]"
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


def extract_metrics(
    model_data: dict[str, Any],
    k_values: list[int],
    include_r_precision: bool = False,
    include_f1_10: bool = False,
    include_mrr: bool = False
) -> dict[str, dict]:
    """Extract precision, recall, and ndcg metrics for specified k values."""
    metrics_dict = model_data.get("metrics", {})

    result = {
        "precision": {},
        "recall": {},
        "ndcg": {},
        "map": metrics_dict.get("map", None)
    }

    for k in k_values:
        result["precision"][k] = metrics_dict.get(f"precision@{k}", None)
        result["recall"][k] = metrics_dict.get(f"recall@{k}", None)
        result["ndcg"][k] = metrics_dict.get(f"ndcg@{k}", None)

    # Add optional metrics
    if include_r_precision:
        result["r_precision"] = metrics_dict.get("r-precision", None)
    if include_f1_10:
        result["f1_10"] = metrics_dict.get("f1@10", None)
    if include_mrr:
        result["mrr"] = metrics_dict.get("mrr", None)

    return result


def generate_typst_table(
    data: dict[str, Any],
    metric_name: str,
    k_shallow: list[int],
    k_deep: list[int],
    table_number: int
) -> str:
    """Generate a Typst table for a specific metric (precision, recall, or ndcg)."""

    metric_key = metric_name.lower()

    # Metric display names
    metric_display = {
        "precision": "P",
        "recall": "R",
        "ndcg": "nDCG"
    }

    metric_symbol = metric_display.get(metric_key, metric_key.upper())

    # Configure columns and extra metrics based on metric type
    # Precision: k=[1,3,5,10,100,1000] + R-Precision (in deep)
    # Recall: k=[1,3,5,10,100,1000] + F1@10 (in shallow)
    # nDCG: k=[1,3,5,10,100,1000] + MRR (in shallow)

    use_r_precision = (metric_key == "precision")
    use_f1_10 = (metric_key == "recall")
    use_mrr = (metric_key == "ndcg")

    # All metrics use k = [1, 3, 5, 10, 100, 1000]
    all_k = k_shallow + k_deep

    # Build headers
    shallow_headers = [f"[{metric_symbol}\\@{k}]" for k in k_shallow]
    deep_headers = [f"[{metric_symbol}\\@{k}]" for k in k_deep]

    # Add extra metric headers
    if use_r_precision:
        deep_headers.append("[R-Prec]")
        num_deep_cols = len(k_deep) + 1
    elif use_f1_10:
        shallow_headers.append("[F1\\@10]")
        num_shallow_cols = len(k_shallow) + 1
        num_deep_cols = len(k_deep)
    elif use_mrr:
        shallow_headers.append("[MRR]")
        num_shallow_cols = len(k_shallow) + 1
        num_deep_cols = len(k_deep)
    else:
        num_shallow_cols = len(k_shallow)
        num_deep_cols = len(k_deep)

    # Fix num_shallow_cols if not set
    if use_r_precision:
        num_shallow_cols = len(k_shallow)

    # Start building the table
    # Calculate total columns
    total_cols = 2 + 1 + num_shallow_cols + num_deep_cols  # Type, Model, MAP, shallow metrics, deep metrics

    lines = [
        "#align(center)[",
        "#text(size: 8.8pt)[",
        "#table(",
        f"      columns: {total_cols},",
        "      column-gutter: 4pt,",
        "      row-gutter: 1pt,",
        "      align: (left, left, center, center, center, center, center, center, center, center),",
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
        f"      table.cell(colspan: {num_shallow_cols}, [*Métricas Rasas* $(k <= 10)$]),",
        f"      table.cell(colspan: {num_deep_cols}, [*Métricas Profundas*]),",
        "",
        f"      table.hline(start: 3, end: {3 + num_shallow_cols}, stroke: 0.3pt),",
        f"      table.hline(start: {3 + num_shallow_cols}, end: {3 + num_shallow_cols + num_deep_cols}, stroke: 0.3pt),",
        "",
        "      // Sub-headers",
    ]

    # Add sub-headers
    lines.append("      " + ", ".join(shallow_headers + deep_headers) + ",")
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

    # First pass: calculate global maximums across all categories
    global_max = {"map": -float('inf')}
    for k in all_k:
        global_max[k] = -float('inf')
    if use_r_precision:
        global_max["r_precision"] = -float('inf')
    if use_f1_10:
        global_max["f1_10"] = -float('inf')
    if use_mrr:
        global_max["mrr"] = -float('inf')

    # Check all BM25 results for global max
    for bm25 in bm25_results:
        metrics = extract_metrics(bm25, all_k, include_r_precision=use_r_precision, include_f1_10=use_f1_10, include_mrr=use_mrr)
        if metrics["map"] is not None and metrics["map"] > global_max["map"]:
            global_max["map"] = metrics["map"]
        for k in all_k:
            val = metrics[metric_key].get(k)
            if val is not None and val > global_max[k]:
                global_max[k] = val
        if use_r_precision and metrics.get("r_precision") is not None and metrics["r_precision"] > global_max["r_precision"]:
            global_max["r_precision"] = metrics["r_precision"]
        if use_f1_10 and metrics.get("f1_10") is not None and metrics["f1_10"] > global_max["f1_10"]:
            global_max["f1_10"] = metrics["f1_10"]
        if use_mrr and metrics.get("mrr") is not None and metrics["mrr"] > global_max["mrr"]:
            global_max["mrr"] = metrics["mrr"]

    # Check all embedding results for global max
    for embedding in embedding_results:
        metrics = extract_metrics(embedding, all_k, include_r_precision=use_r_precision, include_f1_10=use_f1_10, include_mrr=use_mrr)
        if metrics["map"] is not None and metrics["map"] > global_max["map"]:
            global_max["map"] = metrics["map"]
        for k in all_k:
            val = metrics[metric_key].get(k)
            if val is not None and val > global_max[k]:
                global_max[k] = val
        if use_r_precision and metrics.get("r_precision") is not None and metrics["r_precision"] > global_max["r_precision"]:
            global_max["r_precision"] = metrics["r_precision"]
        if use_f1_10 and metrics.get("f1_10") is not None and metrics["f1_10"] > global_max["f1_10"]:
            global_max["f1_10"] = metrics["f1_10"]
        if use_mrr and metrics.get("mrr") is not None and metrics["mrr"] > global_max["mrr"]:
            global_max["mrr"] = metrics["mrr"]

    # Check all fusion results for global max
    for fusion in top_fusions:
        metrics = extract_metrics(fusion, all_k, include_r_precision=use_r_precision, include_f1_10=use_f1_10, include_mrr=use_mrr)
        if metrics["map"] is not None and metrics["map"] > global_max["map"]:
            global_max["map"] = metrics["map"]
        for k in all_k:
            val = metrics[metric_key].get(k)
            if val is not None and val > global_max[k]:
                global_max[k] = val
        if use_r_precision and metrics.get("r_precision") is not None and metrics["r_precision"] > global_max["r_precision"]:
            global_max["r_precision"] = metrics["r_precision"]
        if use_f1_10 and metrics.get("f1_10") is not None and metrics["f1_10"] > global_max["f1_10"]:
            global_max["f1_10"] = metrics["f1_10"]
        if use_mrr and metrics.get("mrr") is not None and metrics["mrr"] > global_max["mrr"]:
            global_max["mrr"] = metrics["mrr"]

    # Calculate max values for BM25 category for each metric
    bm25_max = {"map": -float('inf')}
    for k in all_k:
        bm25_max[k] = -float('inf')
    if use_r_precision:
        bm25_max["r_precision"] = -float('inf')
    if use_f1_10:
        bm25_max["f1_10"] = -float('inf')
    if use_mrr:
        bm25_max["mrr"] = -float('inf')

    for bm25 in bm25_results:
        metrics = extract_metrics(bm25, all_k, include_r_precision=use_r_precision, include_f1_10=use_f1_10, include_mrr=use_mrr)
        if metrics["map"] is not None and metrics["map"] > bm25_max["map"]:
            bm25_max["map"] = metrics["map"]
        for k in all_k:
            val = metrics[metric_key].get(k)
            if val is not None and val > bm25_max[k]:
                bm25_max[k] = val
        if use_r_precision and metrics.get("r_precision") is not None and metrics["r_precision"] > bm25_max["r_precision"]:
            bm25_max["r_precision"] = metrics["r_precision"]
        if use_f1_10 and metrics.get("f1_10") is not None and metrics["f1_10"] > bm25_max["f1_10"]:
            bm25_max["f1_10"] = metrics["f1_10"]
        if use_mrr and metrics.get("mrr") is not None and metrics["mrr"] > bm25_max["mrr"]:
            bm25_max["mrr"] = metrics["mrr"]

    # BM25 models
    lines.append("      // Modelos Esparsos")
    lines.append(f"      table.cell(rowspan: {len(bm25_results)}, [Esparso], align: horizon),")

    for bm25 in bm25_results:
        model_name = get_model_display_name(bm25, "bm25")
        metrics = extract_metrics(bm25, all_k, include_r_precision=use_r_precision, include_f1_10=use_f1_10, include_mrr=use_mrr)

        map_val = format_value(
            metrics["map"],
            is_best_category=(metrics["map"] == bm25_max["map"]),
            is_best_global=(metrics["map"] == global_max["map"])
        )
        values = [
            format_value(
                metrics[metric_key].get(k),
                is_best_category=(metrics[metric_key].get(k) == bm25_max[k]),
                is_best_global=(metrics[metric_key].get(k) == global_max[k])
            )
            for k in all_k
        ]

        if use_r_precision:
            r_prec_val = format_value(
                metrics.get("r_precision"),
                is_best_category=(metrics.get("r_precision") == bm25_max["r_precision"]),
                is_best_global=(metrics.get("r_precision") == global_max["r_precision"])
            )
            values.append(r_prec_val)
        elif use_f1_10:
            f1_val = format_value(
                metrics.get("f1_10"),
                is_best_category=(metrics.get("f1_10") == bm25_max["f1_10"]),
                is_best_global=(metrics.get("f1_10") == global_max["f1_10"])
            )
            values.insert(len(k_shallow), f1_val)  # Insert after shallow k values
        elif use_mrr:
            mrr_val = format_value(
                metrics.get("mrr"),
                is_best_category=(metrics.get("mrr") == bm25_max["mrr"]),
                is_best_global=(metrics.get("mrr") == global_max["mrr"])
            )
            values.insert(len(k_shallow), mrr_val)  # Insert after shallow k values

        lines.append(f"      [{model_name}], [{map_val}], " + ", ".join(f"[{v}]" for v in values) + ",")

    lines.append("")
    lines.append(f"      table.cell(colspan: {total_cols}, []),")
    lines.append("      table.hline(stroke: .6pt),")
    lines.append(f"      table.cell(colspan: {total_cols}, []),")
    lines.append("")

    # Calculate max values for Embedding category for each metric
    embedding_max = {"map": -float('inf')}
    for k in all_k:
        embedding_max[k] = -float('inf')
    if use_r_precision:
        embedding_max["r_precision"] = -float('inf')
    if use_f1_10:
        embedding_max["f1_10"] = -float('inf')
    if use_mrr:
        embedding_max["mrr"] = -float('inf')

    for embedding in embedding_results:
        metrics = extract_metrics(embedding, all_k, include_r_precision=use_r_precision, include_f1_10=use_f1_10, include_mrr=use_mrr)
        if metrics["map"] is not None and metrics["map"] > embedding_max["map"]:
            embedding_max["map"] = metrics["map"]
        for k in all_k:
            val = metrics[metric_key].get(k)
            if val is not None and val > embedding_max[k]:
                embedding_max[k] = val
        if use_r_precision and metrics.get("r_precision") is not None and metrics["r_precision"] > embedding_max["r_precision"]:
            embedding_max["r_precision"] = metrics["r_precision"]
        if use_f1_10 and metrics.get("f1_10") is not None and metrics["f1_10"] > embedding_max["f1_10"]:
            embedding_max["f1_10"] = metrics["f1_10"]
        if use_mrr and metrics.get("mrr") is not None and metrics["mrr"] > embedding_max["mrr"]:
            embedding_max["mrr"] = metrics["mrr"]

    # Embedding models
    lines.append("      // Modelos Semânticos")
    lines.append(f"      table.cell(rowspan: {len(embedding_results)}, [Semântico], align: horizon),")

    for embedding in embedding_results:
        model_name = get_model_display_name(embedding, "embedding")
        metrics = extract_metrics(embedding, all_k, include_r_precision=use_r_precision, include_f1_10=use_f1_10, include_mrr=use_mrr)

        map_val = format_value(
            metrics["map"],
            is_best_category=(metrics["map"] == embedding_max["map"]),
            is_best_global=(metrics["map"] == global_max["map"])
        )
        values = [
            format_value(
                metrics[metric_key].get(k),
                is_best_category=(metrics[metric_key].get(k) == embedding_max[k]),
                is_best_global=(metrics[metric_key].get(k) == global_max[k])
            )
            for k in all_k
        ]

        if use_r_precision:
            r_prec_val = format_value(
                metrics.get("r_precision"),
                is_best_category=(metrics.get("r_precision") == embedding_max["r_precision"]),
                is_best_global=(metrics.get("r_precision") == global_max["r_precision"])
            )
            values.append(r_prec_val)
        elif use_f1_10:
            f1_val = format_value(
                metrics.get("f1_10"),
                is_best_category=(metrics.get("f1_10") == embedding_max["f1_10"]),
                is_best_global=(metrics.get("f1_10") == global_max["f1_10"])
            )
            values.insert(len(k_shallow), f1_val)
        elif use_mrr:
            mrr_val = format_value(
                metrics.get("mrr"),
                is_best_category=(metrics.get("mrr") == embedding_max["mrr"]),
                is_best_global=(metrics.get("mrr") == global_max["mrr"])
            )
            values.insert(len(k_shallow), mrr_val)

        lines.append(f"      [{model_name}], [{map_val}], " + ", ".join(f"[{v}]" for v in values) + ",")

    lines.append("")
    lines.append(f"      table.cell(colspan: {total_cols}, []),")
    lines.append("      table.hline(stroke: .6pt),")
    lines.append(f"      table.cell(colspan: {total_cols}, []),")
    lines.append("")

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

    # Calculate max values for Fusion category for each metric
    fusion_max = {"map": -float('inf')}
    for k in all_k:
        fusion_max[k] = -float('inf')
    if use_r_precision:
        fusion_max["r_precision"] = -float('inf')
    if use_f1_10:
        fusion_max["f1_10"] = -float('inf')
    if use_mrr:
        fusion_max["mrr"] = -float('inf')

    for fusion in top_fusions:
        metrics = extract_metrics(fusion, all_k, include_r_precision=use_r_precision, include_f1_10=use_f1_10, include_mrr=use_mrr)
        if metrics["map"] is not None and metrics["map"] > fusion_max["map"]:
            fusion_max["map"] = metrics["map"]
        for k in all_k:
            val = metrics[metric_key].get(k)
            if val is not None and val > fusion_max[k]:
                fusion_max[k] = val
        if use_r_precision and metrics.get("r_precision") is not None and metrics["r_precision"] > fusion_max["r_precision"]:
            fusion_max["r_precision"] = metrics["r_precision"]
        if use_f1_10 and metrics.get("f1_10") is not None and metrics["f1_10"] > fusion_max["f1_10"]:
            fusion_max["f1_10"] = metrics["f1_10"]
        if use_mrr and metrics.get("mrr") is not None and metrics["mrr"] > fusion_max["mrr"]:
            fusion_max["mrr"] = metrics["mrr"]

    lines.append("      // Fusão")
    lines.append(f"      table.cell(rowspan: {len(top_fusions) + 1}, [Fusão], align: horizon),")

    for fusion in top_fusions:
        model_name = get_model_display_name(fusion, "fusion")
        metrics = extract_metrics(fusion, all_k, include_r_precision=use_r_precision, include_f1_10=use_f1_10, include_mrr=use_mrr)

        map_val = format_value(
            metrics["map"],
            is_best_category=(metrics["map"] == fusion_max["map"]),
            is_best_global=(metrics["map"] == global_max["map"])
        )
        values = [
            format_value(
                metrics[metric_key].get(k),
                is_best_category=(metrics[metric_key].get(k) == fusion_max[k]),
                is_best_global=(metrics[metric_key].get(k) == global_max[k])
            )
            for k in all_k
        ]

        if use_r_precision:
            r_prec_val = format_value(
                metrics.get("r_precision"),
                is_best_category=(metrics.get("r_precision") == fusion_max["r_precision"]),
                is_best_global=(metrics.get("r_precision") == global_max["r_precision"])
            )
            values.append(r_prec_val)
        elif use_f1_10:
            f1_val = format_value(
                metrics.get("f1_10"),
                is_best_category=(metrics.get("f1_10") == fusion_max["f1_10"]),
                is_best_global=(metrics.get("f1_10") == global_max["f1_10"])
            )
            values.insert(len(k_shallow), f1_val)
        elif use_mrr:
            mrr_val = format_value(
                metrics.get("mrr"),
                is_best_category=(metrics.get("mrr") == fusion_max["mrr"]),
                is_best_global=(metrics.get("mrr") == global_max["mrr"])
            )
            values.insert(len(k_shallow), mrr_val)

        lines.append(f"      [{model_name}], [{map_val}], " + ", ".join(f"[{v}]" for v in values) + ",")

    # "E muitos outros algoritmos..." row
    num_value_cols = num_shallow_cols + num_deep_cols
    lines.append("      [E muitos outros algoritmos...], " + ", ".join(["[]"] * num_value_cols) + ",")

    lines.append("")
    lines.append("      table.hline(stroke: .6pt),")
    lines.append("    )")
    lines.append("  ]")
    lines.append("  #v(.4em)")

    # Caption
    metric_name_pt = {
        "precision": "Precisão",
        "recall": "Revocação",
        "ndcg": "nDCG"
    }

    query_group = data.get("query_group")
    if isinstance(query_group, list):
        group_str = ", ".join(str(g) for g in query_group)
    else:
        group_str = str(query_group)

    lines.append(f"  #align(center)[Tabela {table_number}. {metric_name_pt.get(metric_key, metric_key)} para o grupo {group_str}. Valores em *itálico* indicam a melhor métrica dentro de cada categoria (Esparso, Semântico, Fusão). Valores em #rect(fill: rgb(\"#DAA520\"), inset: 1pt)[dourado] indicam a melhor métrica global.]")
    lines.append("]")
    lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_typst_tables.py <json_file> [--output <output_file>]")
        print("Example: python generate_typst_tables.py results/ranx/optimizing_ranx_group3.json")
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

    # Define k values (without 500)
    k_shallow = [1, 3, 5, 10]
    k_deep = [100, 1000]

    # Generate tables
    tables = []

    print("Generating Precision table...")
    precision_table = generate_typst_table(data, "precision", k_shallow, k_deep, 1)
    tables.append(precision_table)

    print("Generating Recall table...")
    recall_table = generate_typst_table(data, "recall", k_shallow, k_deep, 2)
    tables.append(recall_table)

    print("Generating nDCG table...")
    ndcg_table = generate_typst_table(data, "ndcg", k_shallow, k_deep, 3)
    tables.append(ndcg_table)

    # Combine all tables
    full_output = "\n\n".join(tables)

    # Output
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_output)
        print(f"\nTables written to {output_file}")
    else:
        print("\n" + "="*80)
        print("TYPST TABLES OUTPUT")
        print("="*80 + "\n")
        print(full_output)
        print("\n" + "="*80)

        # Auto-save to default location
        default_output = Path(json_file).parent / f"{Path(json_file).stem}_tables.typ"
        with open(default_output, 'w') as f:
            f.write(full_output)
        print(f"\nTables also saved to {default_output}")


if __name__ == "__main__":
    main()
