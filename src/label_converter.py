#!/usr/bin/env python3
"""
label_converter.py
Converte labels binárias (scalar_label/scalar_labels = 0/1) + scalar_field
em classes ABNT NBR 6118 (1–5), replicando exatamente a lógica do PointLabel.html.

Saída: campo `label` (uint8) adicionado/substituído em cada PLY.

Classes:
    1 = Microfissura  (<0.05 mm)
    2 = Fissura       (0.05–0.5 mm)  [apenas no modo sem-labels]
    3 = Trinca        (0.5–1.5 mm)
    4 = Rachadura     (>1.5 mm)
    5 = Normal        (superfície sã + pontos não rotulados)

Modos de conversão por arquivo:
    with_labels  — tem scalar_field + binary labels → autoDetectWithLabels
    no_labels    — tem scalar_field, sem binary labels → autoDetectSegments
    no_sf        — sem scalar_field → crack binário → Microfissura, resto → Normal
    flat_sf      — scalar_field sem variação → mesmo que no_sf
    all_normal   — sem scalar_field e sem labels → tudo Normal
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

BINS = 64  # mesmo valor do PointLabel.html


# ── localização de campos ─────────────────────────────────────────────────────

def _find_field(props: list, candidates: list):
    lower = {p.lower(): p for p in props}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _sf_field(props):
    return _find_field(props, [
        'scalar_Scalar_field', 'scalar_scalar_field',
        'scalar_field',
        'scalar_R', 'scalar_r',
    ])


def _label_field(props):
    # "label" (1–5, já convertido) é excluído intencionalmente
    return _find_field(props, ['scalar_labels', 'scalar_label'])


# ── histograma e suavização ───────────────────────────────────────────────────

def _hist(sf: np.ndarray, mn: float, mx: float) -> np.ndarray:
    rng  = mx - mn or 1.0
    bins = np.clip(((sf - mn) / rng * BINS).astype(int), 0, BINS - 1)
    return np.bincount(bins, minlength=BINS).astype(float)


def _smooth(h: np.ndarray, k: int = 4) -> np.ndarray:
    """Média móvel simétrica — porta de smoothHist() do HTML."""
    out = np.empty_like(h)
    for i in range(len(h)):
        sl = h[max(0, i - k):i + k + 1]
        out[i] = sl.mean()
    return out


# ── modo 1: com binary labels ─────────────────────────────────────────────────

def _classify_with_labels(sf: np.ndarray, binary: np.ndarray,
                           mn: float, mx: float) -> np.ndarray:
    """
    Porta direta de autoDetectWithLabels() do PointLabel.html.

    Divide a região de crack em 3 quantis (33/66):
        SF ≤ q33        → 4 (Rachadura)
        q33 < SF ≤ q66  → 3 (Trinca)
        q66 < SF ≤ thr  → 1 (Microfissura)
        SF > thr        → 5 (Normal)
    """
    rng  = mx - mn or 1.0
    mask = binary > 0

    c_hist = _hist(sf[mask],  mn, mx) if mask.any()  else np.zeros(BINS)
    n_hist = _hist(sf[~mask], mn, mx) if (~mask).any() else np.zeros(BINS)

    crack_bins  = np.where(c_hist > 0)[0]
    normal_bins = np.where(n_hist > 0)[0]

    crack_max_bin  = int(crack_bins[-1])  if len(crack_bins)  else BINS // 2
    normal_min_bin = int(normal_bins[0])  if len(normal_bins) else BINS - 1
    main_thresh    = mn + ((crack_max_bin + normal_min_bin) / 2 / BINS) * rng

    total = c_hist.sum()
    cum   = 0.0
    q33b  = q66b = 0
    for b in range(crack_max_bin + 1):
        cum += c_hist[b]
        if q33b == 0 and cum >= total * 0.33:
            q33b = b
        if q66b == 0 and cum >= total * 0.66:
            q66b = b

    q33 = mn + (q33b / BINS) * rng
    q66 = mn + (q66b / BINS) * rng

    out = np.full(len(sf), 5, dtype=np.uint8)
    out[sf <= main_thresh] = 1  # Microfissura
    out[sf <= q66]         = 3  # Trinca  (sobrescreve Microfissura)
    out[sf <= q33]         = 4  # Rachadura (sobrescreve Trinca)
    return out


# ── modo 2: sem binary labels ─────────────────────────────────────────────────

def _otsu(h: np.ndarray) -> int:
    """Porta de otsuThreshold() do HTML."""
    total = h.sum()
    if total == 0:
        return BINS // 2
    s    = (np.arange(BINS) * h).sum()
    sumB = wB = var_max = t = 0.0
    for i in range(BINS):
        wB  += h[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * h[i]
        v = wB * wF * ((sumB / wB) - (s - sumB) / wF) ** 2
        if v > var_max:
            var_max = v
            t = i
    return int(t)


def _classify_no_labels(sf: np.ndarray, mn: float, mx: float) -> np.ndarray:
    """
    Porta de autoDetectSegments() do HTML (caminho sem binary labels).

    Detecta picos no histograma suavizado, usa vales como thresholds.
    Fallback: Otsu se não houver dois picos distintos.
    Segmentos: [menor SF → maior SF] mapeados para classes 1,2,3,4 + 5 (Normal).
    """
    rng = mx - mn or 1.0
    raw = _hist(sf, mn, mx)
    sm  = _smooth(raw, k=4)
    mx_h     = sm.max() or 1.0
    min_prom = 0.12
    min_dist = max(3, int(BINS * 0.06))

    peaks = []
    for i in range(1, BINS - 1):
        if sm[i] > sm[i - 1] and sm[i] > sm[i + 1] and sm[i] / mx_h >= min_prom:
            if not peaks or i - peaks[-1] >= min_dist:
                peaks.append(i)
            elif sm[i] > sm[peaks[-1]]:
                peaks[-1] = i

    thresh_bins = []
    for p in range(len(peaks) - 1):
        seg = sm[peaks[p]:peaks[p + 1] + 1]
        thresh_bins.append(peaks[p] + int(seg.argmin()))

    if not thresh_bins:
        thresh_bins = [_otsu(raw)]

    thresh_vals = [mn + (b / BINS) * rng for b in thresh_bins]
    crack_seq   = [1, 2, 3, 4]

    # Aplica do maior threshold para o menor: thresholds menores sobrescrevem.
    out = np.full(len(sf), 5, dtype=np.uint8)
    for idx in range(len(thresh_vals) - 1, -1, -1):
        cls = crack_seq[min(idx, len(crack_seq) - 1)]
        out[sf <= thresh_vals[idx]] = cls
    return out


# ── conversão de um arquivo ───────────────────────────────────────────────────

def convert_ply(src: Path, dst: Path) -> dict:
    ply   = PlyData.read(str(src))
    vtx   = ply['vertex']
    props = [p.name for p in vtx.properties]
    n     = len(vtx['x'])

    sf_name  = _sf_field(props)
    lbl_name = _label_field(props)

    sf     = np.array(vtx[sf_name],  dtype=np.float32) if sf_name  else None
    binary = np.array(vtx[lbl_name], dtype=np.uint8)   if lbl_name else None

    if sf is None or (sf.max() - sf.min()) < 0.5:
        # Sem scalar field ou campo flat: crack binário → Microfissura, resto → Normal
        new_labels = np.full(n, 5, dtype=np.uint8)
        if binary is not None:
            new_labels[binary > 0] = 1
        method = 'no_sf' if sf is None else 'flat_sf'

    elif binary is not None and binary.any():
        # Tem labels binárias com pelo menos um crack: subdivide pelo SF
        mn, mx = float(sf.min()), float(sf.max())
        new_labels = _classify_with_labels(sf, binary, mn, mx)
        method = 'with_labels'

    elif binary is not None and not binary.any():
        # Tem label field mas tudo zero (superfície confirmada como normal)
        # Usar peak detection aqui seria incorreto: geraria falsos positivos
        new_labels = np.full(n, 5, dtype=np.uint8)
        method = 'all_normal'

    else:
        # Sem label field: detecta classes pela distribuição do SF
        mn, mx = float(sf.min()), float(sf.max())
        new_labels = _classify_no_labels(sf, mn, mx)
        method = 'no_labels'

    # Reconstrói vertex preservando todos os campos originais + label
    names, dtypes_list, arrays = [], [], []
    for p in vtx.properties:
        if p.name == 'label':
            continue  # substituído abaixo
        arr = np.array(vtx[p.name])
        names.append(p.name)
        dtypes_list.append((p.name, arr.dtype))
        arrays.append(arr)

    names.append('label')
    dtypes_list.append(('label', np.dtype('u1')))
    arrays.append(new_labels)

    rec = np.rec.fromarrays(arrays, dtype=dtypes_list)
    el  = PlyElement.describe(rec, 'vertex')
    dst.parent.mkdir(parents=True, exist_ok=True)
    PlyData([el], byte_order='<').write(str(dst))

    counts = {c: int((new_labels == c).sum()) for c in range(1, 6)}
    return {'file': src.name, 'method': method, **counts}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Converte scalar_label(s) + scalar_field → label ABNT NBR 6118 (1–5)'
    )
    ap.add_argument('--input',   default='data',      help='Diretório raiz com os PLYs originais')
    ap.add_argument('--output',  default='data_abnt', help='Diretório de saída (ignorado com --inplace)')
    ap.add_argument('--inplace', action='store_true', help='Sobrescreve os PLYs originais')
    ap.add_argument('--glob',    default='**/*.ply',  help='Padrão glob dentro de --input')
    ap.add_argument('--dry-run', action='store_true', help='Mostra o que seria feito sem escrever nada')
    args = ap.parse_args()

    src_root = Path(args.input).resolve()
    dst_root = src_root if args.inplace else Path(args.output).resolve()

    files = sorted(src_root.glob(args.glob))
    if not files:
        sys.exit(f'Nenhum .ply encontrado em {src_root}/{args.glob}')

    CLASS_NAMES = {1: 'Micro', 2: 'Fissura', 3: 'Trinca', 4: 'Rachadura', 5: 'Normal'}

    print(f'\n{"="*82}')
    print(f'  LABEL CONVERTER — ABNT NBR 6118')
    print(f'  {len(files)} arquivos  |  {src_root}  →  {dst_root}')
    if args.dry_run:
        print('  [DRY RUN — nenhum arquivo será escrito]')
    print(f'{"="*82}')
    hdr = f'  {"arquivo":42s} {"método":14s}'
    for c in range(1, 6):
        hdr += f'  {CLASS_NAMES[c]:>9}'
    print(hdr)
    print(f'  {"-"*80}')

    totals   = {c: 0 for c in range(1, 6)}
    by_method = {}
    errors   = []

    for src in files:
        rel = src.relative_to(src_root)
        dst = src if args.inplace else dst_root / rel

        try:
            if args.dry_run:
                # apenas detecta campos, não escreve
                ply   = PlyData.read(str(src))
                props = [p.name for p in ply['vertex'].properties]
                sf    = _sf_field(props)
                lb    = _label_field(props)
                method = ('with_labels' if sf and lb else
                          'no_labels'   if sf       else
                          'no_sf')
                row = {'file': src.name, 'method': method,
                       **{c: -1 for c in range(1, 6)}}
            else:
                row = convert_ply(src, dst)

            method = row['method']
            by_method[method] = by_method.get(method, 0) + 1

            line = f'  {src.name:42s} [{method:12s}]'
            for c in range(1, 6):
                v = row[c]
                line += f'  {v:>9,}' if v >= 0 else f'  {"—":>9}'
            print(line)

            if not args.dry_run:
                for c in range(1, 6):
                    totals[c] += row[c]

        except Exception as exc:
            errors.append((src.name, str(exc)))
            print(f'  {"ERRO":42s}  {src.name}: {exc}')

    print(f'\n  {"TOTAL":42s} {"":14s}', end='')
    for c in range(1, 6):
        print(f'  {totals[c]:>9,}', end='')
    print()

    print(f'\n  Métodos: ' + ' | '.join(f'{m}={n}' for m, n in sorted(by_method.items())))
    if errors:
        print(f'\n  {len(errors)} ERROS:')
        for fname, msg in errors:
            print(f'    {fname}: {msg}')
    else:
        print(f'\n  {len(files)} arquivos convertidos sem erros → {dst_root}')


if __name__ == '__main__':
    main()
