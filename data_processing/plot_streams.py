################################################################################
# Plot n data streams on one axis in the shared PAI figure style from a spec.  #
################################################################################

#
"""
Imports
"""
import os
import json
import shutil
import argparse
import pai_style
import matplotlib

matplotlib.use('Agg')

import os.path as osp
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from typing           import Any, Dict, List, Optional, Tuple

#
"""
Config
"""
pai_style.apply_rc_style()

# Graph padding on the x-axis and y-axis
# Override the x-axis padding with --x-from-zero
y_pad_frac = 0.18
x_pad_frac = 0.15

default_x_label = 'Parameters (millions)'
default_y_fmt   = '.4f'

# Default label placement, (dx points, dy points, ha, va). Line stream
# labels sit above their points, scatter labels below, the anchor label
# to its left. Any point may override this with its own offset
line_offset    = (0, 9, 'center', 'bottom')
scatter_offset = (0, -9, 'center', 'top')
anchor_offset  = (-12, 0, 'right', 'center')

Offset = Tuple[int, int, str, str]

#
"""
Functions
"""
def load_spec(path: str) -> Dict[str, Any]:
    '''
    Read the spec and raise early if a required field is missing

    Signature:
        path (str):
            - Path of the JSON spec
    '''
    with open(path) as f:
        spec = json.load(f)
    if 'out' not in spec:
        raise ValueError('spec needs an "out" filename')
    if 'y_axis' not in spec or 'label' not in spec['y_axis']:
        raise ValueError('spec needs y_axis.label, the y axis is required')
    if not spec.get('streams'):
        raise ValueError('spec needs at least one stream')
    for i, stream in enumerate(spec['streams']):
        if 'name' not in stream:
            raise ValueError(f'stream {i} has no name')
        if stream.get('style', 'line') not in ('line', 'scatter'):
            raise ValueError(
                f'stream {stream["name"]} style must be line or scatter')
        if not stream.get('points'):
            raise ValueError(f'stream {stream["name"]} has no points')
    return spec

def parse_point(raw: Any) -> Dict[str, Any]:
    '''
    Normalize a spec point into x, y, label, and offset

    Notes:
        - A point can be a bare [x, y] pair or an object with x, y, and
          optional label and offset
        - offset is [dx, dy, ha, va] in points. None means the stream
          default placement

    Signature:
        raw (Any):
            - Point as written in the spec
    '''
    if isinstance(raw, (list, tuple)):
        return dict(x=raw[0], y=raw[1], label=None, offset=None)
    offset = raw.get('offset')
    return dict(
        x      = raw['x'],
        y      = raw['y'],
        label  = raw.get('label'),
        offset = tuple(offset) if offset else None,
    )

def scale_x(value: float, custom_x: bool) -> float:
    # Default x is a raw parameter count plotted in millions
    return value if custom_x else value / 1e6

def annotate_point(
    ax     : plt.Axes,
    x      : float,
    y      : float,
    text   : str,
    offset : Offset,
) -> None:
    '''
    Write some text next to a point

    Signature:
        ax (plt.Axes):
            - The axes the point is on
        x (float):
            - Point x in data coordinates
        y (float):
            - Point y in data coordinates
        text (str):
            - What to write
        offset (Offset):
            - Nudge in points, then the horizontal and vertical alignment
    '''
    dx, dy, ha, va = offset
    ax.annotate(
        text,
        (x, y),
        xytext      = (dx, dy),
        textcoords  = 'offset points',
        ha          = ha,
        va          = va,
        fontsize    = 9,
        color       = pai_style.neutral_ink,
        linespacing = 1.4,
    )

def point_text(point: Dict[str, Any], fmt: str) -> str:
    # Value alone, or the point's name on a line above it
    value = format(point['y'], fmt)
    return f'{point["label"]}\n{value}' if point['label'] else value

def draw_anchor(
    ax       : plt.Axes,
    anchor   : Dict[str, Any],
    custom_x : bool,
    annotate : bool,
    fmt      : str,
) -> Tuple[float, float]:
    '''
    Draw the hollow anchor marker and return where it landed

    Signature:
        ax (plt.Axes):
            - The axes to draw on
        anchor (Dict[str, Any]):
            - The anchor entry from the spec
        custom_x (bool):
            - True when x is not a parameter count and passes through as is
        annotate (bool):
            - Whether to write the value beside the marker
        fmt (str):
            - Format spec for the value, .4f by default
    '''
    point = parse_point(anchor)
    x, y  = scale_x(point['x'], custom_x), point['y']
    ax.plot(
        x, y,
        marker          = 'o',
        markersize      = 10,
        markerfacecolor = 'white',
        markeredgecolor = pai_style.neutral_ink,
        markeredgewidth = 2,
        linestyle       = 'none',
        zorder          = 4,
    )
    if annotate:
        annotate_point(
            ax, x, y, point_text(point, fmt), point['offset'] or anchor_offset)
    return x, y

def draw_stream(
    ax       : plt.Axes,
    stream   : Dict[str, Any],
    color    : str,
    anchor   : Optional[Tuple[float, float]],
    custom_x : bool,
    annotate : bool,
    fmt      : str,
) -> Tuple[List[float], List[float]]:
    '''
    Draw one stream and return the x and y values it covers

    Notes:
        - A line stream joins its points in spec order. If there is an
          anchor the line starts there, unless from_anchor is false
        - A scatter stream is just markers. Set dash_to_anchor and each
          one gets a dashed line back to the anchor

    Signature:
        ax (plt.Axes):
            - The axes to draw on
        stream (Dict[str, Any]):
            - The stream entry from the spec
        color (str):
            - Hex color for the markers and line
        anchor (Optional[Tuple[float, float]]):
            - Where the anchor was drawn, or None if the spec has none
        custom_x (bool):
            - True when x is not a parameter count and passes through as is
        annotate (bool):
            - Whether to write the value beside every point
        fmt (str):
            - Format spec for the values, .4f by default
    '''
    points = [parse_point(p) for p in stream['points']]
    xs     = [scale_x(p['x'], custom_x) for p in points]
    ys     = [p['y'] for p in points]
    style  = stream.get('style', 'line')

    if style == 'line':
        lx, ly = list(xs), list(ys)
        if anchor is not None and stream.get('from_anchor', True):
            lx, ly = [anchor[0]] + lx, [anchor[1]] + ly
        ax.plot(lx, ly, linestyle='-', linewidth=2, color=color, zorder=2)
    elif anchor is not None and stream.get('dash_to_anchor', False):
        for x, y in zip(xs, ys):
            ax.plot(
                [anchor[0], x], [anchor[1], y],
                linestyle = '--',
                linewidth = 1.5,
                color     = color,
                zorder    = 2,
            )

    ax.plot(
        xs, ys,
        marker          = 'o',
        markersize      = 10,
        markerfacecolor = color,
        markeredgecolor = color,
        linestyle       = 'none',
        zorder          = 3,
    )
    if annotate:
        default = line_offset if style == 'line' else scatter_offset
        for point, x, y in zip(points, xs, ys):
            annotate_point(
                ax, x, y, point_text(point, fmt), point['offset'] or default)
    return xs, ys

def legend_handle(name: str, color: str, style: str) -> Line2D:
    '''
    Make a legend entry that looks like the stream it stands for

    Signature:
        name (str):
            - Text shown in the legend
        color (str):
            - The stream color, or neutral ink for the anchor
        style (str):
            - line, scatter, or anchor
    '''
    hollow = style == 'anchor'
    return Line2D(
        [], [],
        marker          = 'o',
        markersize      = 9,
        markerfacecolor = 'white' if hollow else color,
        markeredgecolor = color,
        markeredgewidth = 2,
        color           = color,
        linewidth       = 2,
        linestyle       = '-' if style == 'line' else 'none',
        label           = name,
    )

def window_axes(
    ax          : plt.Axes,
    xs          : List[float],
    ys          : List[float],
    y_lim       : Optional[List[float]],
    x_from_zero : bool,
) -> None:
    '''
    Set the axis limits around the data with some breathing room

    Notes:
        - If the data has no span, a single point or a flat line, we pad
          on the value itself so the axis still has width

    Signature:
        ax (plt.Axes):
            - The axes to window
        xs (List[float]):
            - Every x we plotted
        ys (List[float]):
            - Every y we plotted
        y_lim (Optional[List[float]]):
            - An explicit [lo, hi] from the spec, which wins over the window
        x_from_zero (bool):
            - Start the x axis at zero instead of windowing on the data
    '''
    if y_lim:
        ax.set_ylim(*y_lim)
    else:
        span = max(ys) - min(ys) or abs(max(ys)) * 0.05 or 1.0
        ax.set_ylim(min(ys) - span * y_pad_frac, max(ys) + span * y_pad_frac)
    if x_from_zero:
        ax.set_xlim(0, max(xs) * (1 + x_pad_frac))
    else:
        span = max(xs) - min(xs) or abs(max(xs)) * 0.05 or 1.0
        ax.set_xlim(min(xs) - span * x_pad_frac, max(xs) + span * x_pad_frac)

def plot(spec: Dict[str, Any], x_from_zero: bool) -> plt.Figure:
    '''
    Draw the whole figure the spec describes

    Signature:
        spec (Dict[str, Any]):
            - The spec, already loaded and checked
        x_from_zero (bool):
            - Start the x axis at zero instead of windowing on the data
    '''
    custom_x = 'x_axis' in spec
    x_label  = spec['x_axis']['label'] if custom_x else default_x_label
    y_axis   = spec['y_axis']
    fmt      = y_axis.get('format', default_y_fmt)
    annotate = spec.get('annotate', True)
    streams  = spec['streams']

    fig, ax = plt.subplots(figsize=(9, 6.5))
    all_xs, all_ys, handles = [], [], []

    anchor_xy = None
    if spec.get('anchor'):
        anchor_xy = draw_anchor(
            ax, spec['anchor'], custom_x, annotate, fmt)
        all_xs.append(anchor_xy[0])
        all_ys.append(anchor_xy[1])
        handles.append(legend_handle(
            spec['anchor'].get('name', 'Vanilla'),
            pai_style.neutral_ink,
            'anchor',
        ))

    for i, stream in enumerate(streams):
        color  = stream.get('color') or pai_style.stream_color(i, len(streams))
        xs, ys = draw_stream(
            ax, stream, color, anchor_xy, custom_x, annotate, fmt)
        all_xs.extend(xs)
        all_ys.extend(ys)
        handles.append(
            legend_handle(stream['name'], color, stream.get('style', 'line')))

    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_axis['label'], labelpad=10)
    if spec.get('title'):
        ax.set_title(spec['title'], pad=18)

    window_axes(ax, all_xs, all_ys, y_axis.get('lim'), x_from_zero)
    pai_style.apply_axes_style(ax)
    ax.legend(handles=handles, loc='best', **pai_style.legend_kwargs)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Plot n data streams on one axis in the shared PAI '
                      'figure style from a JSON spec'
    )
    parser.add_argument(
        'spec',
        type = str,
        help = 'path of the JSON spec describing the figure',
    )
    parser.add_argument(
        '--out-dir',
        type    = str,
        default = '',
        help    = 'directory for the PNG and a copy of the spec, '
                  'defaults to the directory holding the spec',
    )
    parser.add_argument(
        '--x-from-zero',
        action = 'store_true',
        help   = 'anchor the x axis at zero instead of windowing on the data',
    )
    args = parser.parse_args()

    spec_path = osp.abspath(args.spec)
    out_dir   = osp.dirname(spec_path)
    if args.out_dir:
        out_dir = osp.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    spec = load_spec(spec_path)
    fig  = plot(spec, args.x_from_zero)
    stem = osp.splitext(osp.basename(spec['out']))[0]
    png  = osp.join(out_dir, f'{stem}.png')
    pai_style.save_figure(fig, png)
    print(f'wrote {png}')

    # Keep the spec beside the PNG so the figure carries its own numbers
    spec_copy = osp.join(out_dir, f'{stem}.json')
    if osp.realpath(spec_copy) != osp.realpath(spec_path):
        shutil.copyfile(spec_path, spec_copy)
