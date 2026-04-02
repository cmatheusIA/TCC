"""
Visualizador de nuvens de pontos salvas em visualizations_ts/.

Uso:
  uv run python src/open_ply.py             # lista os arquivos e abre o mais recente
  uv run python src/open_ply.py --all       # abre todos em sequência
  uv run python src/open_ply.py --idx 2     # abre o arquivo de índice 2 da lista
  uv run python src/open_ply.py --file avaria_5_avarias_01042026_2244.ply
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pathlib import Path
from plyfile import PlyData
import open3d as o3d


VIS_PATH = Path(__file__).resolve().parents[1] / "visualizations_ts"


def load_ply_as_pcd(path: Path) -> o3d.geometry.PointCloud:
    data   = PlyData.read(str(path)).elements[0].data
    xyz    = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float64)
    colors = np.stack([data['red'], data['green'], data['blue']], axis=1).astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def list_plys() -> list[Path]:
    files = sorted(VIS_PATH.glob("*.ply"), key=lambda p: p.stat().st_mtime)
    return files


def show(path: Path):
    print(f"\nAbrindo: {path.name}")
    pcd = load_ply_as_pcd(path)
    print(f"  Pontos : {len(pcd.points):,}")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=path.stem,
        width=1280,
        height=720,
        point_show_normal=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualizador de PLYs — visualizations_ts/")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--all",  action="store_true", help="Abre todos os PLYs em sequência")
    group.add_argument("--idx",  type=int, metavar="N", help="Abre pelo índice da lista")
    group.add_argument("--file", type=str, metavar="NOME", help="Abre pelo nome do arquivo")
    args = parser.parse_args()

    files = list_plys()
    if not files:
        print(f"Nenhum arquivo .ply encontrado em:\n  {VIS_PATH}")
        sys.exit(1)

    print(f"\nArquivos em {VIS_PATH.name}/")
    print("-" * 50)
    for i, f in enumerate(files):
        marker = "← mais recente" if i == len(files) - 1 else ""
        print(f"  [{i:2d}]  {f.name}  {marker}")
    print()

    if args.all:
        for f in files:
            show(f)

    elif args.idx is not None:
        if args.idx < 0 or args.idx >= len(files):
            print(f"Índice {args.idx} fora do intervalo [0, {len(files)-1}]")
            sys.exit(1)
        show(files[args.idx])

    elif args.file:
        path = VIS_PATH / args.file
        if not path.exists():
            # tenta correspondência parcial
            matches = [f for f in files if args.file in f.name]
            if not matches:
                print(f"Arquivo não encontrado: {args.file}")
                sys.exit(1)
            path = matches[-1]
        show(path)

    else:
        # padrão: abre o mais recente
        show(files[-1])


if __name__ == "__main__":
    main()
