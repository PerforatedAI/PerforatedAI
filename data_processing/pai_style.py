################################################################################
# Shared colors, axes styling, and figure saving for every PAI plot.           #
################################################################################

#
"""
Imports
"""
import colorsys

import matplotlib.pyplot as plt

from matplotlib.colors import to_hex, to_rgb
from typing            import Dict

#
"""
Config
"""
# Stream colors in order
# Use rainbow colors after these initial 4 colors
stream_palette = ['#00FAC9', '#B0B7C3', '#1B669F', '#DD8452']

rainbow_hue_start  = 0.8
rainbow_hue_span   = 0.75
rainbow_saturation = 0.8
rainbow_value      = 0.9

# Neutral ink for the anchor edge, point labels, and the legend frame
neutral_ink = '#333333'

# How far the darkest shade drops when one hue is split across dendrites
max_shade_drop = 0.35

# Keyword arguments every legend call passes through
legend_kwargs: Dict[str, object] = dict(
    frameon    = True,
    framealpha = 1.0,
    edgecolor  = neutral_ink,
    fontsize   = 10,
)

figure_dpi = 200

#
"""
Functions
"""
def apply_rc_style() -> None:
    # Font sizes shared by every figure, called once per script at import
    plt.rcParams.update({
        'font.size'       : 10,
        'axes.titlesize'  : 15,
        'axes.labelsize'  : 12,
        'xtick.labelsize' : 10,
        'ytick.labelsize' : 10,
        'legend.fontsize' : 10,
        'figure.titlesize': 15,
    })

def stream_color(index: int, total: int) -> str:
    '''
    Pick the color for one stream out of total streams

    Notes:
        - The first four streams take the fixed palette in order
        - After that we spread hues evenly over rainbow_hue_span, so a
          figure with six streams gets two extra hues a third of a
          wheel apart

    Signature:
        index (int):
            - Where this stream sits in the figure, counting from zero
        total (int):
            - How many streams the figure has
    '''
    if index < len(stream_palette):
        return stream_palette[index]
    extra = max(1, total - len(stream_palette))
    step  = rainbow_hue_span / extra
    hue   = (rainbow_hue_start + (index - len(stream_palette)) * step) % 1.0
    rgb   = colorsys.hsv_to_rgb(hue, rainbow_saturation, rainbow_value)
    return to_hex(rgb)

def shade_color(color: str, fraction: float) -> str:
    '''
    Darken a color a little, to step dendrite counts within one model

    Notes:
        - fraction 0 leaves the color alone and fraction 1 gives the
          darkest shade, max_shade_drop below the original

    Signature:
        color (str):
            - Any color matplotlib understands
        fraction (float):
            - How far along the ramp to go, from 0 to 1
    '''
    h, s, v = colorsys.rgb_to_hsv(*to_rgb(color))
    v       = v * (1.0 - max_shade_drop * fraction)
    return to_hex(colorsys.hsv_to_rgb(h, s, v))

def apply_axes_style(ax: plt.Axes) -> None:
    '''
    Apply the shared grid and spine settings to an axes

    Notes:
        - Dotted grid drawn underneath the data, top and right spines
          hidden, so every figure in the repo looks the same

    Signature:
        ax (plt.Axes):
            - The axes to style
    '''
    ax.grid(linestyle=':', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

def save_figure(fig: plt.Figure, path: str) -> None:
    '''
    Write a figure to disk at the shared dpi and close it

    Notes:
        - The bounding box is tight, so labels that spill past the axes
          still make it into the PNG

    Signature:
        fig (plt.Figure):
            - The figure to write
        path (str):
            - Where the PNG goes
    '''
    fig.savefig(path, dpi=figure_dpi, bbox_inches='tight')
    plt.close(fig)
