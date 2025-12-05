# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///

import json
import matplotlib.pyplot as plt
import numpy as np
import sys

# Get group number(s) from CLI argument, default to 1
# Supports single group (1) or multiple groups (1,2,3)
# Also supports res parameter: res=0 (default) or res=1
# Also supports hiper parameter: hiper=0 (default) or hiper=1
# Also supports pr parameter: pr=0 (default) or pr=1 (precision-recall curve)
group = 1
res = 0
hiper = 0
pr = 0

if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        if arg.startswith('res='):
            res = int(arg.split('=')[1])
        elif arg.startswith('hiper='):
            hiper = int(arg.split('=')[1])
        elif arg.startswith('pr='):
            pr = int(arg.split('=')[1])
        elif ',' in arg:
            # Multiple groups like "1,2,3"
            groups = [int(g.strip()) for g in arg.split(',')]
            group = ''.join(str(g) for g in groups)
        else:
            # Single group
            try:
                group = int(arg)
            except ValueError:
                pass  # Ignore invalid arguments

# Read the JSON file
with open(f'results/complete_results/optimizing_group{group}.json', 'r') as f:
    data = json.load(f)

# Print structure summary
print("=" * 80)
print("JSON STRUCTURE SUMMARY")
print("=" * 80)
print(f"\nQuery Group: {data['query_group']} - {data['query_group_name']}")
print("\n--- Optimization Summary ---")
for key, value in data['optimization_summary'].items():
    print(f"  {key}: {value}")

print("\n--- Best Results ---")
print(f"  Best Embedding: {data['best_embedding']['variant']} (MAP: {data['best_embedding']['metrics']['map']:.4f})")
print(f"  Best BM25: {data['best_bm25']['variant']} (MAP: {data['best_bm25']['metrics']['map']:.4f})")
print(f"  Best Fusion: {data['best_fusion']['fusion_method']} (MAP: {data['best_fusion']['metrics']['map']:.4f})")

# Extract embedding results
embeddings = []
for emb in data['all_embeddings_results']:
    if emb.get('metrics') is not None:
        embeddings.append({
            'variant': emb['variant'],
            'map': emb['metrics']['map'],
            'ndcg@10': emb['metrics']['ndcg@10'],
            'mrr@10': emb['metrics']['mrr@10'],
            'precision@10': emb['metrics']['precision@10']
        })

# Extract BM25 results
bm25_results = []
for bm25 in data['all_bm25_results']:
    if bm25.get('metrics') is not None:
        bm25_results.append({
            'variant': bm25['variant'],
            'config': f"k1={bm25.get('k1', 'N/A')}, b={bm25.get('b', 'N/A')}",
            'map': bm25['metrics']['map'],
            'ndcg@10': bm25['metrics']['ndcg@10'],
            'mrr@10': bm25['metrics']['mrr@10']
        })

# Extract fusion results (for hyperparameter optimization visualization)
fusion_results = []
if 'all_fusion_results' in data:
    for fusion in data['all_fusion_results']:
        if fusion.get('metrics') is not None:
            fusion_results.append({
                'fusion_method': fusion.get('fusion_method', 'unknown'),
                'fusion_norm': fusion.get('fusion_norm', 'unknown'),
                'fusion_k': fusion.get('fusion_k', 60),
                'map': fusion['metrics'].get('map', 0),
                'ndcg@10': fusion['metrics'].get('ndcg@10', 0),
                'mrr@10': fusion['metrics'].get('mrr@10', 0),
                'precision@10': fusion['metrics'].get('precision@10', 0)
            })

# Sort by MAP score
embeddings.sort(key=lambda x: x['map'], reverse=True)
bm25_results.sort(key=lambda x: x['map'], reverse=True)
fusion_results.sort(key=lambda x: x['map'], reverse=True)

print(f"\n--- All Embedding Variants ({len(embeddings)} successful) ---")
for emb in embeddings:
    print(f"  {emb['variant']:20s} MAP: {emb['map']:.4f}, NDCG@10: {emb['ndcg@10']:.4f}")

print(f"\n--- All BM25 Variants ({len(bm25_results)} configs) ---")
for i, bm25 in enumerate(bm25_results[:10]):  # Show top 10
    print(f"  {bm25['variant']:15s} {bm25['config']:20s} MAP: {bm25['map']:.4f}")

if fusion_results:
    print(f"\n--- All Fusion Variants ({len(fusion_results)} configs) ---")
    for i, fusion in enumerate(fusion_results[:10]):  # Show top 10
        print(f"  {fusion['fusion_method']:15s} norm={fusion['fusion_norm']:10s} k={fusion['fusion_k']:3d} MAP: {fusion['map']:.4f}")

# Helper function to extract interpolated precision-recall if available
def extract_interp_pr(metrics_dict):
    """Extract 11-point interpolated precision-recall from metrics."""
    interp_pr = {}
    for key, value in metrics_dict.items():
        if key.startswith('interp_p@'):
            recall_level = float(key.split('@')[1])
            interp_pr[recall_level] = value
    return interp_pr

# Create visualizations with transparent background
if pr == 1:
    # Single plot for precision-recall curve comparison
    fig, ax_pr = plt.subplots(1, 1, figsize=(12, 10), facecolor='none')
    ax_pr.set_facecolor('none')
    ax_pr.patch.set_alpha(0)

    fig.suptitle(f'Curva Precisão-Revocação Interpolada (11 pontos) - {data["query_group_name"]} (Group {data["query_group"]})',
                 fontsize=16, fontweight='bold', y=0.98)

    # Extract interpolated P-R curves for each method
    bm25_pr = extract_interp_pr(data['best_bm25']['metrics'])
    embedding_pr = extract_interp_pr(data['best_embedding']['metrics'])
    fusion_pr = extract_interp_pr(data['best_fusion']['metrics'])

    if not bm25_pr or not embedding_pr or not fusion_pr:
        print("\n" + "="*80)
        print("WARNING: Interpolated precision-recall data not found in results!")
        print("Please run benchmarks with --compute_interp_pr True")
        print("="*80)
        axes = None
    else:
        # Standard recall levels
        recall_levels = sorted(bm25_pr.keys())

        # Extract precision values
        bm25_precisions = [bm25_pr[r] for r in recall_levels]
        embedding_precisions = [embedding_pr[r] for r in recall_levels]
        fusion_precisions = [fusion_pr[r] for r in recall_levels]

        # Plot curves
        ax_pr.plot(recall_levels, bm25_precisions, marker='o', linewidth=3, markersize=8,
                   label=f"BM25 ({data['best_bm25']['variant']})", color='#e67e22', alpha=0.9)
        ax_pr.plot(recall_levels, embedding_precisions, marker='s', linewidth=3, markersize=8,
                   label=f"Embedding ({data['best_embedding']['variant']})", color='#3498db', alpha=0.9)
        ax_pr.plot(recall_levels, fusion_precisions, marker='^', linewidth=3, markersize=8,
                   label=f"Fusion ({data['best_fusion']['fusion_method']})", color='#2ecc71', alpha=0.9)

        ax_pr.set_xlabel('Revocação (Recall)', fontweight='bold', fontsize=14)
        ax_pr.set_ylabel('Precisão (Precision)', fontweight='bold', fontsize=14)
        ax_pr.set_title('Comparação de Desempenho: Precisão vs Revocação',
                       fontweight='bold', fontsize=14, pad=20)
        ax_pr.set_xlim(-0.05, 1.05)
        ax_pr.set_ylim(0, 1.05)
        ax_pr.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax_pr.legend(loc='upper right', fontsize=12, framealpha=0.9)

        # Add annotations for key points
        # Annotate at recall 0.5 (mid-point)
        mid_idx = 5  # Index for recall 0.5
        for method, precisions, color, marker in [
            ('BM25', bm25_precisions, '#e67e22', 'o'),
            ('Embedding', embedding_precisions, '#3498db', 's'),
            ('Fusion', fusion_precisions, '#2ecc71', '^')
        ]:
            ax_pr.annotate(f'{precisions[mid_idx]:.3f}',
                          xy=(recall_levels[mid_idx], precisions[mid_idx]),
                          xytext=(10, -10), textcoords='offset points',
                          fontsize=9, color=color, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color))

        # Add area under curve (approximate)
        bm25_auc = np.trapz(bm25_precisions, recall_levels)
        embedding_auc = np.trapz(embedding_precisions, recall_levels)
        fusion_auc = np.trapz(fusion_precisions, recall_levels)

        # Add text box with AUC values
        textstr = f'Área sob a curva (AUC):\n'
        textstr += f'BM25: {bm25_auc:.3f}\n'
        textstr += f'Embedding: {embedding_auc:.3f}\n'
        textstr += f'Fusion: {fusion_auc:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2)
        ax_pr.text(0.02, 0.02, textstr, transform=ax_pr.transAxes, fontsize=11,
                  verticalalignment='bottom', horizontalalignment='left', bbox=props,
                  fontweight='bold')

    axes = None  # No axes array in pr mode

elif hiper == 1:
    # Single plot for hyperparameter optimization
    fig, ax_hiper = plt.subplots(1, 1, figsize=(18, 10), facecolor='none')
    ax_hiper.set_facecolor('none')
    ax_hiper.patch.set_alpha(0)

    fig.suptitle(f'Otimização de Hiperparâmetros - Fusion - {data["query_group_name"]} (Group {data["query_group"]})',
                 fontsize=18, fontweight='bold', y=0.98)

    # Plot hyperparameter optimization for fusion
    if fusion_results:
        # Filter out results with MAP = 0 or N/A
        fusion_filtered = [f for f in fusion_results if f['map'] > 0]

        # Remove duplicates - keep only unique configurations based on method, norm, and k
        seen = set()
        fusion_unique = []
        for f in fusion_filtered:
            key = (f['fusion_method'], f['fusion_norm'], f['fusion_k'], round(f['map'], 6))
            if key not in seen:
                seen.add(key)
                fusion_unique.append(f)

        if not fusion_unique:
            print("WARNING: No valid fusion results with MAP > 0")
            axes = None
        else:
            # Get all fusion results sorted by MAP
            fusion_sorted = sorted(fusion_unique, key=lambda x: x['map'])

            maps = [f['map'] for f in fusion_sorted]

            # Color gradient from worst (red) to best (green) - VIBRANT COLORS
            colors = []
            n = len(fusion_sorted)
            for i in range(n):
                # Gradient from red to orange to yellow to light green to green
                ratio = i / (n - 1) if n > 1 else 0

                if ratio < 0.33:
                    # Red to Orange (0.0 - 0.33)
                    local_ratio = ratio / 0.33
                    r = 1.0
                    g = 0.3 + (0.4 * local_ratio)  # 0.3 -> 0.7
                    b = 0.0
                elif ratio < 0.67:
                    # Orange to Yellow (0.33 - 0.67)
                    local_ratio = (ratio - 0.33) / 0.34
                    r = 1.0
                    g = 0.7 + (0.3 * local_ratio)  # 0.7 -> 1.0
                    b = 0.0
                else:
                    # Yellow to Green (0.67 - 1.0)
                    local_ratio = (ratio - 0.67) / 0.33
                    r = 1.0 - (0.8 * local_ratio)  # 1.0 -> 0.2
                    g = 1.0
                    b = 0.2 * local_ratio  # 0.0 -> 0.2

                colors.append((r, g, b, 0.9))

            # Highlight best and worst configurations with solid colors
            if n > 0:
                colors[-1] = '#27ae60'  # Darker green for best
                colors[0] = '#c0392b'   # Darker red for worst

            y_positions = np.arange(len(fusion_sorted))
            bars = ax_hiper.barh(y_positions, maps, color=colors, edgecolor='#2c3e50', linewidth=0.5)

            # Remove Y-axis labels to focus on visual growth
            ax_hiper.set_yticks([])
            ax_hiper.set_ylabel('Configurações (Pior → Melhor)', fontweight='bold', fontsize=14)
            ax_hiper.set_xlabel('MAP Score', fontweight='bold', fontsize=16)
            ax_hiper.set_title('Otimização de Hiperparâmetros: Evolução do Desempenho',
                              fontweight='bold', fontsize=16, pad=20)
            ax_hiper.set_xlim(0, max(maps) * 1.1)
            ax_hiper.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)

            # Add annotations only for best and worst
            worst_idx = 0
            best_idx = len(maps) - 1

            # Worst annotation
            ax_hiper.annotate(f'Pior: {maps[worst_idx]:.4f}\n{fusion_sorted[worst_idx]["fusion_method"]}',
                             xy=(maps[worst_idx], worst_idx),
                             xytext=(max(maps)*0.3, worst_idx),
                             fontsize=11, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='#e74c3c', alpha=0.8, edgecolor='black'),
                             color='white',
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black', lw=2))

            # Best annotation
            ax_hiper.annotate(f'Melhor: {maps[best_idx]:.4f}\n{fusion_sorted[best_idx]["fusion_method"]}',
                             xy=(maps[best_idx], best_idx),
                             xytext=(max(maps)*0.3, best_idx),
                             fontsize=11, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='#27ae60', alpha=0.8, edgecolor='black'),
                             color='white',
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black', lw=2))

            # Add improvement annotation
            worst_map = maps[0]
            best_map = maps[-1]

            # Calculate improvement
            improvement = ((best_map - worst_map) / worst_map) * 100
            improvement_text = f'+{improvement:.1f}%'

            # Add text box with improvement statistics
            textstr = f'Pior: {worst_map:.4f}\nMelhor: {best_map:.4f}\nMelhoria: {improvement_text}\nTotal configs: {len(fusion_sorted)}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2)
            ax_hiper.text(0.98, 0.02, textstr, transform=ax_hiper.transAxes, fontsize=12,
                         verticalalignment='bottom', horizontalalignment='right', bbox=props,
                         fontweight='bold')

    axes = None  # No axes array in hiper mode

elif res == 1:
    # Only 2 plots side by side
    fig, axes_temp = plt.subplots(1, 2, figsize=(16, 6), facecolor='none')
    # Convert to 2D array to maintain compatibility
    axes = np.array([[axes_temp[0], axes_temp[1], None],
                     [None, None, None]], dtype=object)
    fig.suptitle(f'Optimization Results - {data["query_group_name"]} (Group {data["query_group"]})',
                 fontsize=16, fontweight='bold')
else:
    # Original 6 plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='none')
    fig.suptitle(f'Optimization Results - {data["query_group_name"]} (Group {data["query_group"]})',
                 fontsize=16, fontweight='bold')

# Set transparent background for all axes
if axes is not None:
    for ax in axes.flat:
        if ax is not None:
            ax.set_facecolor('none')
            ax.patch.set_alpha(0)

# Only create regular plots if not in hiper mode
if hiper == 0 and axes is not None:
    # 1. Embedding Variants Comparison
    ax1 = axes[0, 0]
    variants = [e['variant'] for e in embeddings]
    maps = [e['map'] for e in embeddings]
    colors = ['#2ecc71' if v == data['best_embedding']['variant'] else '#3498db' for v in variants]

    bars = ax1.barh(variants, maps, color=colors)
    ax1.set_xlabel('MAP Score', fontweight='bold')
    ax1.set_title('Embedding Variants Performance (by MAP)', fontweight='bold')
    ax1.set_xlim(0, max(maps) * 1.1)
    for i, (bar, score) in enumerate(zip(bars, maps)):
        ax1.text(score + 0.005, i, f'{score:.4f}', va='center', fontsize=9)
    ax1.grid(axis='x', alpha=0.3)

    # 2. Multiple Metrics for Top Embeddings OR Detailed Metrics Comparison (based on res parameter)
    ax2 = axes[0, 1]

    if res == 1:
        # Show detailed metrics comparison (same as position [0, 2])
        metrics_names = ['MAP', 'NDCG@10', 'MRR@10', 'P@10']
        bm25_metrics = [
            data['best_bm25']['metrics']['map'],
            data['best_bm25']['metrics']['ndcg@10'],
            data['best_bm25']['metrics']['mrr@10'],
            data['best_bm25']['metrics']['precision@10']
        ]
        embedding_metrics = [
            data['best_embedding']['metrics']['map'],
            data['best_embedding']['metrics']['ndcg@10'],
            data['best_embedding']['metrics']['mrr@10'],
            data['best_embedding']['metrics']['precision@10']
        ]
        fusion_metrics = [
            data['best_fusion']['metrics']['map'],
            data['best_fusion']['metrics']['ndcg@10'],
            data['best_fusion']['metrics']['mrr@10'],
            data['best_fusion']['metrics']['precision@10']
        ]
    
        x = np.arange(len(metrics_names))
        width = 0.25
    
        bars1 = ax2.bar(x - width, bm25_metrics, width, label=f"BM25 ({data['best_bm25']['variant']})",
                        color='#e67e22', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax2.bar(x, embedding_metrics, width, label=f"Embedding ({data['best_embedding']['variant']})",
                        color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
        bars3 = ax2.bar(x + width, fusion_metrics, width, label=f"Fusion ({data['best_fusion']['fusion_method']})",
                        color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
    
        ax2.set_ylabel('Score', fontweight='bold', fontsize=11)
        ax2.set_title('Comparação de Métricas: Melhores Configurações', fontweight='bold', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_names, fontsize=10)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, max(max(bm25_metrics), max(embedding_metrics), max(fusion_metrics)) * 1.15)
    
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=7, rotation=0)
    else:
        # Original: Top 5 Embeddings
        top_embeddings = embeddings[:5]
        x = np.arange(len(top_embeddings))
        width = 0.2
    
        metrics_to_plot = ['map', 'ndcg@10', 'mrr@10', 'precision@10']
        metric_labels = ['MAP', 'NDCG@10', 'MRR@10', 'P@10']
        colors_metrics = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6']
    
        for i, (metric, label, color) in enumerate(zip(metrics_to_plot, metric_labels, colors_metrics)):
            values = [e[metric] for e in top_embeddings]
            ax2.bar(x + i*width, values, width, label=label, color=color, alpha=0.8)
    
        ax2.set_xlabel('Embedding Variant', fontweight='bold')
        ax2.set_ylabel('Score', fontweight='bold')
        ax2.set_title('Top 5 Embeddings - Multiple Metrics', fontweight='bold')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels([e['variant'] for e in top_embeddings], rotation=45, ha='right')
        ax2.legend(loc='upper right')
        ax2.grid(axis='y', alpha=0.3)

    # 3. Top BM25 Configurations (only shown when res=0)
    if res == 0:
        ax3 = axes[1, 0]
        top_bm25 = bm25_results[:10]
        bm25_labels = [f"{b['variant']}\n{b['config']}" for b in top_bm25]
        bm25_maps = [b['map'] for b in top_bm25]
        colors_bm25 = ['#2ecc71' if b['variant'] == data['best_bm25']['variant'] and
                       b['config'] == f"k1={data['best_bm25']['k1']}, b={data['best_bm25']['b']}"
                       else '#e67e22' for b in top_bm25]
    
        bars = ax3.barh(range(len(top_bm25)), bm25_maps, color=colors_bm25)
        ax3.set_yticks(range(len(top_bm25)))
        ax3.set_yticklabels(bm25_labels, fontsize=8)
        ax3.set_xlabel('MAP Score', fontweight='bold')
        ax3.set_title('Top 10 BM25 Configurations', fontweight='bold')
        ax3.set_xlim(0, max(bm25_maps) * 1.1)
        for i, (bar, score) in enumerate(zip(bars, bm25_maps)):
            ax3.text(score + 0.005, i, f'{score:.4f}', va='center', fontsize=8)
        ax3.grid(axis='x', alpha=0.3)
    
    # 4. Overall Best Results Comparison (only shown when res=0)
    if res == 0:
        ax4 = axes[1, 1]
        comparison_data = {
            'Best\nEmbedding': data['best_embedding']['metrics']['map'],
            'Best\nBM25': data['best_bm25']['metrics']['map'],
            'Best\nFusion': data['best_fusion']['metrics']['map']
        }
    
        bars = ax4.bar(comparison_data.keys(), comparison_data.values(),
                       color=['#3498db', '#e67e22', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=2)
        ax4.set_ylabel('MAP Score', fontweight='bold')
        ax4.set_title('Best Configuration Comparison', fontweight='bold')
        ax4.set_ylim(0, max(comparison_data.values()) * 1.2)
    
        for bar, (name, value) in zip(bars, comparison_data.items()):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{value:.4f}',
                     ha='center', va='bottom', fontweight='bold', fontsize=12)
    
            # Add method details
            if 'Embedding' in name:
                detail = data['best_embedding']['variant']
            elif 'BM25' in name:
                detail = f"{data['best_bm25']['variant']}"
            else:
                detail = f"{data['best_fusion']['fusion_method']}"
    
            ax4.text(bar.get_x() + bar.get_width()/2., height/2,
                     detail,
                     ha='center', va='center', fontsize=9, style='italic')
    
        ax4.grid(axis='y', alpha=0.3)
    
    # 5. Detailed Metrics Comparison (MAP, NDCG@10, MRR@10, P@10) (only shown when res=0)
    if res == 0:
        ax5 = axes[0, 2]
        metrics_names = ['MAP', 'NDCG@10', 'MRR@10', 'P@10']
        bm25_metrics = [
            data['best_bm25']['metrics']['map'],
            data['best_bm25']['metrics']['ndcg@10'],
            data['best_bm25']['metrics']['mrr@10'],
            data['best_bm25']['metrics']['precision@10']
        ]
        embedding_metrics = [
            data['best_embedding']['metrics']['map'],
            data['best_embedding']['metrics']['ndcg@10'],
            data['best_embedding']['metrics']['mrr@10'],
            data['best_embedding']['metrics']['precision@10']
        ]
        fusion_metrics = [
            data['best_fusion']['metrics']['map'],
            data['best_fusion']['metrics']['ndcg@10'],
            data['best_fusion']['metrics']['mrr@10'],
            data['best_fusion']['metrics']['precision@10']
        ]
    
        x = np.arange(len(metrics_names))
        width = 0.25
    
        bars1 = ax5.bar(x - width, bm25_metrics, width, label=f"BM25 ({data['best_bm25']['variant']})",
                        color='#e67e22', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax5.bar(x, embedding_metrics, width, label=f"Embedding ({data['best_embedding']['variant']})",
                        color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
        bars3 = ax5.bar(x + width, fusion_metrics, width, label=f"Fusion ({data['best_fusion']['fusion_method']})",
                        color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
    
        ax5.set_ylabel('Score', fontweight='bold', fontsize=11)
        ax5.set_title('Comparação de Métricas: Melhores Configurações', fontweight='bold', fontsize=12)
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics_names, fontsize=10)
        ax5.legend(loc='upper right', fontsize=9)
        ax5.grid(axis='y', alpha=0.3)
        ax5.set_ylim(0, max(max(bm25_metrics), max(embedding_metrics), max(fusion_metrics)) * 1.15)
    
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=7, rotation=0)
    
    # 6. Growth percentage visualization (only shown when res=0)
    if res == 0:
        ax6 = axes[1, 2]
        bm25_map = data['best_bm25']['metrics']['map']
        embedding_map = data['best_embedding']['metrics']['map']
        fusion_map = data['best_fusion']['metrics']['map']
    
        embedding_growth = ((embedding_map - bm25_map) / bm25_map) * 100
        fusion_growth = ((fusion_map - bm25_map) / bm25_map) * 100
    
        methods = ['Embedding\nvs BM25', 'Fusion\nvs BM25']
        growths = [embedding_growth, fusion_growth]
        colors_growth = ['#3498db', '#2ecc71']
    
        bars = ax6.bar(methods, growths, color=colors_growth, alpha=0.8, edgecolor='black', linewidth=2)
        ax6.set_ylabel('Crescimento (%)', fontweight='bold', fontsize=11)
        ax6.set_title('Porcentagem de Melhoria vs BM25', fontweight='bold', fontsize=12)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax6.set_ylim(0, max(growths) * 1.2)
        ax6.grid(axis='y', alpha=0.3)
    
        for bar, value in zip(bars, growths):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'+{value:.1f}%',
                     ha='center', va='bottom', fontweight='bold', fontsize=13)

plt.tight_layout()

# Determine output filename based on mode parameter
if pr == 1:
    output_file = f'results/complete_results/optimizing_group{group}_precision_recall.png'
    mode_desc = '11-Point Interpolated Precision-Recall Curve'
elif hiper == 1:
    output_file = f'results/complete_results/optimizing_group{group}_hyperparameter.png'
    mode_desc = 'Hyperparameter Optimization (Fusion)'
elif res == 1:
    output_file = f'results/complete_results/optimizing_group{group}_visualization_res.png'
    mode_desc = 'Detailed Metrics Comparison'
else:
    output_file = f'results/complete_results/optimizing_group{group}_visualization.png'
    mode_desc = 'Full Optimization Overview'

# Save with maximum quality, transparent background
plt.savefig(output_file,
            dpi=600,                    # High DPI for maximum quality
            bbox_inches='tight',
            transparent=True,           # Transparent background
            facecolor='none',           # No face color
            edgecolor='none',           # No edge color
            format='png',
            metadata={'Software': 'matplotlib'})
print(f"\n{'='*80}")
print(f"Visualization saved to: {output_file}")
print(f"High quality PNG with transparent background (600 DPI)")
print(f"Mode: {mode_desc}")
print(f"{'='*80}")
plt.show()
