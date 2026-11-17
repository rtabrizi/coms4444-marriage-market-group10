"""Combine the two market-ranking tables into a single vector PDF figure.

Reads the checked aggregate CSV tables; does not run any simulations.
Requires ReportLab and Matplotlib. Run from any working directory.
"""

import csv
import math
from pathlib import Path

from matplotlib.font_manager import FontProperties, findfont
from reportlab.lib.colors import Color, HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / 'output/pdf/player10_tables'
OUTPUT = ROOT / 'output/pdf/player10_market_rankings_combined.pdf'
MARKETS = ['equal', 'equal+random', 'half-half', 'half-half_sample',
           'solo+random', '90random_10group', 'overall']
HEADERS = ['Equal\ngroups', 'Equal +\nrandom', 'Half-and-half',
           'Half-and-half\n(sampled)', 'Group +\nrandom', '90% random', 'Overall']
INK = HexColor('#183249')
GRAY = HexColor('#CBD5E1')
BLUE = (29 / 255, 78 / 255, 216 / 255)
COLORS = [tuple(channel * (1 - i / 9) + i / 9 for channel in BLUE)
          for i in range(10)]
PAGE_W, PAGE_H = 896, 832
LEFT, RIGHT = 86, 12
CELL_W = (PAGE_W - LEFT - RIGHT) / 7
ROW_H = 34


def text_color(rgb):
    linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in rgb]
    luminance = sum(v * weight for v, weight in zip(linear, (.2126, .7152, .0722)))
    return white if luminance < .235 else INK


def load_table(filename, metric):
    with (TABLES / filename).open(newline='') as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 70
    lookup = {(row['roster'], int(row['group'])): row for row in rows}
    assert len(lookup) == 70
    for market in MARKETS:
        ranked = sorted(range(1, 11), key=lambda g: float(lookup[market, g][metric]), reverse=True)
        assert [int(lookup[market, g]['rank']) for g in ranked] == list(range(1, 11))
    for group in range(1, 11):
        parts = [lookup[market, group] for market in MARKETS[:-1]]
        overall = lookup['overall', group]
        seats = sum(int(row['seats']) for row in parts)
        assert seats == int(overall['seats'])
        for numerator, rate in [('positive', 'positive_rate'), ('happiness', 'mean_happiness')]:
            pooled = sum(float(row[numerator]) for row in parts) / seats
            assert math.isclose(pooled, float(overall[rate]), rel_tol=0, abs_tol=1e-10)
    return lookup


def draw_table(pdf, lookup, top, metric):
    metric_label = 'Positive outcomes (%)' if metric == 'positive_rate' else 'Mean happiness'
    pdf.saveState()
    pdf.translate(22, top - 5 * ROW_H)
    pdf.rotate(90)
    pdf.setFillColor(INK)
    pdf.setFont('Figure-Bold', 14)
    pdf.drawCentredString(0, 0, metric_label)
    pdf.restoreState()
    pdf.setFillColor(INK)
    pdf.setFont('Figure-Bold', 12)
    for col, heading in enumerate(HEADERS):
        lines = heading.split('\n')
        for line, value in enumerate(lines):
            pdf.drawCentredString(LEFT + (col + .5) * CELL_W,
                                  top + 16 + 14 * (len(lines) - 1 - line), value)
    for row, group in enumerate(range(1, 11)):
        bottom = top - (row + 1) * ROW_H
        pdf.setFillColor(INK)
        pdf.setFont('Figure-Bold' if group == 10 else 'Figure', 12)
        pdf.drawRightString(LEFT - 12, bottom + 12, f'P{group}')
        for col, market in enumerate(MARKETS):
            data = lookup[market, group]
            rank = int(data['rank'])
            rgb = COLORS[rank - 1]
            x = LEFT + col * CELL_W
            pdf.setFillColor(Color(*rgb))
            pdf.rect(x + 2, bottom + 1, CELL_W - 4, ROW_H - 2, stroke=0, fill=1)
            pdf.setFillColor(text_color(rgb))
            pdf.setFont('Figure-Bold', 13)
            pdf.drawCentredString(x + CELL_W / 2, bottom + 19, f'#{rank}')
            value = float(data[metric])
            label = f'{100 * value:.2f}%' if metric == 'positive_rate' else f'{value:.2f}'
            pdf.setFont('Figure', 11.5)
            pdf.drawCentredString(x + CELL_W / 2, bottom + 5, label)
        if group == 10:
            pdf.setStrokeColor(INK)
            pdf.setLineWidth(1.3)
            pdf.rect(LEFT + .5, bottom + .5, 7 * CELL_W - 1, ROW_H - 1, stroke=1, fill=0)
    pdf.setStrokeColor(GRAY)
    pdf.setLineWidth(.65)
    pdf.line(LEFT + 6 * CELL_W, top + 37, LEFT + 6 * CELL_W, top - 10 * ROW_H)


def main():
    positive = load_table('market_rankings.csv', 'positive_rate')
    happiness = load_table('market_rankings_mean_happiness.csv', 'mean_happiness')
    for key in positive:
        assert positive[key]['seats'] == happiness[key]['seats']
        assert positive[key]['positive_rate'] == happiness[key]['positive_rate']
        assert positive[key]['mean_happiness'] == happiness[key]['mean_happiness']
    pdfmetrics.registerFont(TTFont('Figure', findfont(FontProperties(family='DejaVu Sans'))))
    pdfmetrics.registerFont(TTFont('Figure-Bold', findfont(FontProperties(family='DejaVu Sans', weight='bold'))))
    pdf = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    pdf.setTitle('Player rankings by market type')
    draw_table(pdf, positive, top=784, metric='positive_rate')
    draw_table(pdf, happiness, top=388, metric='mean_happiness')
    # One shared, discrete scale: exact white at rank 10 and blue at rank 1.
    legend_w = 34
    legend_x = (PAGE_W - 10 * legend_w) / 2
    legend_y = 14
    for i, rgb in enumerate(COLORS):
        pdf.setFillColor(Color(*rgb))
        pdf.rect(legend_x + i * legend_w, legend_y, legend_w - 1, 18, stroke=0, fill=1)
        pdf.setFillColor(text_color(rgb))
        pdf.setFont('Figure', 10)
        pdf.drawCentredString(legend_x + (i + .5) * legend_w, legend_y + 5, str(i + 1))
    pdf.setFillColor(INK)
    pdf.setFont('Figure', 10)
    pdf.drawRightString(legend_x - 8, legend_y + 5, 'Best')
    pdf.drawString(legend_x + 10 * legend_w + 8, legend_y + 5, 'Worst')
    pdf.showPage()
    pdf.save()
    print(OUTPUT)


if __name__ == '__main__':
    main()
