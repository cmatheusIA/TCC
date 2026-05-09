"""
Visualization utilities shared across all detector scripts.

  save_crack_ply   — PLY com pontos crack em vermelho, restante em cor original
  plot_loco_metrics — gráficos 2D de métricas por nuvem + resumo
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement


def save_crack_ply(
    xyz: np.ndarray,
    rgb_orig: np.ndarray,
    pred_labels: np.ndarray,
    out_path: str,
    gt_labels: np.ndarray | None = None,
) -> None:
    """
    Salva dois arquivos PLY com pontos de crack em vermelho.

      <out_path>          — predições do modelo: crack=vermelho, restante=cor original
      <base>_gt<ext>      — ground-truth:         crack=vermelho, restante=cor original
                            (só gerado se gt_labels fornecido)

    Args:
        xyz        : (N, 3) float32
        rgb_orig   : (N, 3) float32 em [0, 1]
        pred_labels: (N,)  int/bool — 1 = predito como crack
        out_path   : caminho de saída para o PLY de predições
        gt_labels  : (N,)  int/bool opcional — 1 = crack real (GT)
    """
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    N = len(xyz)
    dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32),
             ('red', np.uint8), ('green', np.uint8), ('blue', np.uint8)]

    def _write(colors: np.ndarray, path: str) -> None:
        v = np.zeros(N, dtype=dtype)
        v['x'] = xyz[:, 0]; v['y'] = xyz[:, 1]; v['z'] = xyz[:, 2]
        v['red'] = colors[:, 0]; v['green'] = colors[:, 1]; v['blue'] = colors[:, 2]
        PlyData([PlyElement.describe(v, 'vertex')], text=False).write(path)

    # PLY de predições — crack em vermelho
    c_pred = (np.clip(rgb_orig, 0.0, 1.0) * 255).astype(np.uint8)
    c_pred[pred_labels.astype(bool)] = [255, 0, 0]
    _write(c_pred, out_path)

    # PLY de ground-truth — crack em vermelho (opcional)
    if gt_labels is not None:
        base, ext = os.path.splitext(out_path)
        c_gt = (np.clip(rgb_orig, 0.0, 1.0) * 255).astype(np.uint8)
        c_gt[gt_labels.astype(bool)] = [255, 0, 0]
        _write(c_gt, base + '_gt' + ext)


def plot_loco_metrics(
    results: list,
    out_dir: str,
    model_name: str,
    ts: str = '',
) -> None:
    """
    Salva dois gráficos 2D de métricas para avaliação LOCO.

      per_cloud_metrics[_ts].png  — barras agrupadas (AUROC / F1 / AP) por nuvem
      summary_metrics[_ts].png    — barras com média ± std resumidas

    Args:
        results   : lista de dicts com chaves 'filename', 'auroc', 'f1', 'ap'
        out_dir   : diretório de saída
        model_name: rótulo para títulos dos gráficos
        ts        : sufixo de timestamp (opcional)
    """
    if not results:
        return
    os.makedirs(out_dir, exist_ok=True)

    sfx    = f'_{ts}' if ts else ''
    fnames = [r['filename'].replace('.ply', '') for r in results]
    aurocs = np.array([r['auroc'] for r in results], dtype=float)
    f1s    = np.array([r['f1']    for r in results], dtype=float)
    aps    = np.array([r.get('ap', r.get('avg_precision', 0.0)) for r in results], dtype=float)
    n      = len(results)
    x      = np.arange(n)
    w      = 0.25

    # ── 1. Gráfico de barras agrupado por nuvem ───────────────────────────────
    fig, ax = plt.subplots(figsize=(max(9, n * 0.9 + 2), 5))
    ax.bar(x - w, aurocs, w, label='AUROC', color='#2196F3', alpha=0.85)
    ax.bar(x,     f1s,    w, label='F1',    color='#4CAF50', alpha=0.85)
    ax.bar(x + w, aps,    w, label='AP',    color='#FF9800', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(fnames, rotation=45, ha='right', fontsize=7)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score')
    ax.set_title(f'{model_name} — Métricas por Nuvem (LOCO)')
    ax.axhline(0.5, color='red', linestyle='--', lw=0.8, alpha=0.5, label='Aleatório')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(out_dir, f'per_cloud_metrics{sfx}.png')
    fig.savefig(p1, dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ── 2. Resumo com barras de erro (média ± std) ────────────────────────────
    labels_m = ['AUROC', 'F1', 'AP']
    means    = [float(aurocs.mean()), float(f1s.mean()), float(aps.mean())]
    stds     = [float(aurocs.std()),  float(f1s.std()),  float(aps.std())]
    colors   = ['#2196F3', '#4CAF50', '#FF9800']

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    bars = ax2.bar(labels_m, means, color=colors, alpha=0.85,
                   yerr=stds, capsize=7, error_kw={'linewidth': 1.5})
    for bar, m, s in zip(bars, means, stds):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.015,
            f'{m:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold',
        )
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel('Score')
    ax2.set_title(f'{model_name} — Resumo LOCO  (n={n})')
    ax2.axhline(0.5, color='red', linestyle='--', lw=0.8, alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(out_dir, f'summary_metrics{sfx}.png')
    fig2.savefig(p2, dpi=130, bbox_inches='tight')
    plt.close(fig2)

    print(f"  Visualizações 2D: {p1}")
    print(f"                    {p2}")
