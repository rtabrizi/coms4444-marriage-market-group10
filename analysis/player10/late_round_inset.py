"""Add a final-quarter detail view to a cumulative outcome chart."""
from matplotlib.ticker import MaxNLocator
import numpy as np


def add_late_round_inset(ax, bounds=(.43, .16, .53, .35)):
    detail = ax.inset_axes(bounds)
    for line in ax.lines:
        x, y = line.get_data()
        late = np.asarray(x) >= 75
        copy, = detail.plot(x[late], y[late])
        copy.update_from(line)
        copy.set_transform(detail.transData)
        copy.set_clip_path(detail.patch)
        copy.set_linewidth(1.4)
    detail.set(xlim=(75, 100), xticks=[75, 100])
    detail.yaxis.set_major_locator(MaxNLocator(nbins=3))
    detail.tick_params(labelsize=8, length=2, pad=1)
    detail.set_title('Final quarter', fontsize=8, pad=2)
    detail.grid(axis='y', linewidth=.5, color='#E2E8F0')
    for spine in detail.spines.values():
        spine.set_visible(True)
        spine.set_color('#94A3B8')
