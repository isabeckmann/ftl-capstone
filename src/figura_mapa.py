"""Figura do artigo: mapa da bacia do Taquari-Antas até Encantado.

Mostra as três sub-bacias usadas para a chuva (Antas, Taquari superior, Guaporé), o contorno da bacia de Encantado,
as 4 estações da ANA, a grade de 0,1° do IMERG e, opcionalmente, a chuva acumulada do IMERG Early num período
(padrão: 29/04 a 05/05/2024, a semana da cheia de maio de 2024) como fundo.

Polígonos: src/basin.py (HydroBASINS nível 12) -> data/processed/bacias_estacoes.gpkg, recortados em sub-bacias sem
sobreposição por src.imerg_early.subbasin_geoms. Chuva: grades meia-horárias em data/processed/imerg_early/.

Uso:
    python -m src.figura_mapa                                   # com chuva de fundo (29/04 a 05/05/2024)
    python -m src.figura_mapa --sem-chuva
    python -m src.figura_mapa --inicio 2023-11-15 --fim 2023-11-19 --saida outputs/figs/mapa_nov2023.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from . import config  # noqa: E402
from . import imerg_early as ie  # noqa: E402
from .basin import STATION_META  # noqa: E402

SAIDA = config.OUTPUTS / "figs" / "mapa_bacia.png"

# Três primeiras cores categóricas da paleta de referência (validadas para 3 categorias, inclusive daltonismo)
SUBBACIAS = [("antas", "Antas", "#2a78d6"),
             ("taquari_sup", "Taquari superior", "#eb6834"),
             ("guapore", "Guaporé", "#1baf7a")]
TINTA, TINTA_2, GRADE = "#1f1f1d", "#52514c", "#9a998f"
# (dx, dy) do rótulo de cada estação, em pontos, para não sobrepor as vizinhas
ROTULOS = {"86472000": ("Linha José Júlio", (6, 5), "left"),
           "86472600": ("Santa Tereza", (6, -11), "left"),
           "86510000": ("Muçum", (-7, 4), "right"),
           "86720000": ("Encantado", (-7, -10), "right")}


def chuva_acumulada(inicio, fim, grid_dir=ie.GRID_DIR) -> xr.DataArray:
    """Soma (mm) das meias-horas do IMERG Early com início entre `inicio` 00:00 e `fim` 23:30 (UTC)."""
    t0, t1 = pd.Timestamp(inicio), pd.Timestamp(fim) + pd.Timedelta(hours=23, minutes=30)
    meses = pd.period_range(t0, t1, freq="M")
    arqs = [Path(grid_dir) / f"imerg_early_{m:%Y-%m}.nc" for m in meses]
    falta = [a.name for a in arqs if not a.exists()]
    if falta:
        raise FileNotFoundError(f"Faltam grades do IMERG Early: {falta}. Rode python -m src.imerg_early baixar ...")
    partes = []
    for a in arqs:
        with xr.open_dataset(a) as ds:
            partes.append(ds["precipitation"].sel(time=slice(t0, t1)).load())
    da = xr.concat(partes, dim="time")
    n_esp = int((t1 - t0) / pd.Timedelta(minutes=30)) + 1
    if da.sizes["time"] < 0.95 * n_esp:
        print(f"[figura_mapa] aviso: só {da.sizes['time']} de {n_esp} meias-horas no período")
    return (da * 0.5).sum("time")            # mm/h em passos de 30 min -> mm


def _area_km2(geom) -> float:
    from pyproj import Geod

    return abs(Geod(ellps="WGS84").geometry_area_perimeter(geom)[0]) / 1e6


def _graus(eixo):
    def f(v, _):
        s = ("O" if v < 0 else "L") if eixo == "x" else ("S" if v < 0 else "N")
        return f"{abs(v):.1f}°{s}".replace(".", ",")
    return FuncFormatter(f)


def _desenha_poligono(ax, geom, **kw):
    """Desenha (Multi)Polygon do shapely com matplotlib, sem depender do geopandas.plot."""
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch

    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for p in polys:
        verts, codes = [], []
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            verts += xy.tolist()
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2) + [MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), **kw))


def figura(saida=SAIDA, com_chuva=True, inicio="2024-04-29", fim="2024-05-05", dpi=300):
    geoms = ie.subbasin_geoms()
    x0, y0, x1, y1 = ie.BBOX
    lat_media = (y0 + y1) / 2

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": TINTA_2,
                         "axes.linewidth": 0.6, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_facecolor("white")

    # fundo: chuva acumulada (cinza, para não competir com as cores das sub-bacias)
    cb = None
    if com_chuva:
        acc = chuva_acumulada(inicio, fim)
        lon_b = np.r_[acc.lon.values - 0.05, acc.lon.values[-1] + 0.05]
        lat_b = np.r_[acc.lat.values - 0.05, acc.lat.values[-1] + 0.05]
        vmax = float(np.nanpercentile(acc.values, 99))
        im = ax.pcolormesh(lon_b, lat_b, acc.values, cmap="Greys", vmin=0, vmax=vmax, shading="flat",
                           alpha=0.7, zorder=0, rasterized=True)
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, extend="max")
        cb.set_label(f"Chuva acumulada, IMERG Early (mm)\n{pd.Timestamp(inicio):%d/%m} a {pd.Timestamp(fim):%d/%m/%Y} (UTC)",
                     fontsize=7.5, color=TINTA_2)
        cb.ax.tick_params(labelsize=7, color=TINTA_2)
        cb.outline.set_linewidth(0.4)

    # sub-bacias
    alfa = 0.22 if com_chuva else 0.55           # com chuva de fundo, o preenchimento é leve e o contorno identifica
    for k, _, cor in SUBBACIAS:
        _desenha_poligono(ax, geoms[k], facecolor=cor, alpha=alfa, edgecolor="none", zorder=2)
        _desenha_poligono(ax, geoms[k], facecolor="none", edgecolor=cor, linewidth=2.0 if com_chuva else 1.1,
                          zorder=3)
    _desenha_poligono(ax, geoms["bacia"], facecolor="none", edgecolor=TINTA, linewidth=1.8, zorder=4,
                      joinstyle="round")

    # grade IMERG 0,1° (bordas das células)
    for x in np.arange(x0, x1 + 1e-9, 0.1):
        ax.axvline(x, color=GRADE, linewidth=0.25, alpha=0.7, zorder=1)
    for y in np.arange(y0, y1 + 1e-9, 0.1):
        ax.axhline(y, color=GRADE, linewidth=0.25, alpha=0.7, zorder=1)

    # estações da ANA
    halo = [patheffects.withStroke(linewidth=2.6, foreground="white")]
    for code, meta in STATION_META.items():
        nome, (dx, dy), ha = ROTULOS[code]
        alvo = code in config.TARGETS
        ax.plot(meta["lon"], meta["lat"], marker="^" if alvo else "o", markersize=7 if alvo else 5.5,
                markerfacecolor=TINTA if alvo else "white", markeredgecolor=TINTA if alvo else TINTA,
                markeredgewidth=1.0, zorder=6, linestyle="none")
        ax.annotate(f"{nome}\n{code}", (meta["lon"], meta["lat"]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=7, color=TINTA, path_effects=halo, zorder=7, linespacing=1.0)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1 / np.cos(np.deg2rad(lat_media)))
    ax.xaxis.set_major_formatter(_graus("x"))
    ax.yaxis.set_major_formatter(_graus("y"))
    ax.set_xticks(np.arange(-52.5, x1, 0.5))
    ax.set_yticks(np.arange(-30.0, y1, 0.5))
    ax.tick_params(length=3, width=0.5)
    ax.set_title("Bacia do Taquari-Antas até Encantado", loc="left", fontsize=11, fontweight="bold", color=TINTA,
                 pad=8)

    itens = [Patch(facecolor=cor, alpha=max(alfa, 0.45), edgecolor=cor, linewidth=1.1,
                   label=f"{nome} ({_area_km2(geoms[k]):,.0f} km²)".replace(",", "."))
             for k, nome, cor in SUBBACIAS]
    itens += [Line2D([], [], color=TINTA, linewidth=1.8,
                     label=f"Bacia de Encantado ({_area_km2(geoms['bacia']):,.0f} km²)".replace(",", ".")),
              Line2D([], [], marker="^", color=TINTA, linestyle="none", markersize=7, label="Estação ANA (alvo)"),
              Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=TINTA, color=TINTA,
                     linestyle="none", markersize=5.5, label="Estação ANA (montante)"),
              Line2D([], [], color=GRADE, linewidth=0.6, label="Grade IMERG 0,1°")]
    leg = ax.legend(handles=itens, loc="lower right", fontsize=7, frameon=True, framealpha=0.92,
                    edgecolor="#c3c2b7", borderpad=0.7, handlelength=1.8)
    leg.get_frame().set_linewidth(0.5)
    fig.text(0.01, 0.01, "Fontes: HydroBASINS v1c (Lehner & Grill, 2013); NASA GPM IMERG Early V07; ANA.",
             fontsize=6.5, color=TINTA_2)

    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return saida


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", default=str(SAIDA))
    ap.add_argument("--sem-chuva", action="store_true", help="não desenha a chuva acumulada de fundo")
    ap.add_argument("--inicio", default="2024-04-29")
    ap.add_argument("--fim", default="2024-05-05")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    p = figura(a.saida, not a.sem_chuva, a.inicio, a.fim, a.dpi)
    print(f"Salvo: {p}")
