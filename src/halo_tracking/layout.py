"""Reusable 3D spacing measured relative to the rendered axes area."""
import itertools
import weakref
import numpy as np
from matplotlib.artist import Artist
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, FixedLocator
from matplotlib.transforms import IdentityTransform, Bbox
from mpl_toolkits.mplot3d import proj3d


class _Relative3DLayout(Artist):
    """Reserve space and choose tick density before each axes render."""

    def __init__(self, ax, gap_fraction):
        super().__init__()
        self.target = weakref.ref(ax)
        self.gap_fraction = gap_fraction
        self.aspect = np.asarray(ax.get_box_aspect()).copy()
        self.set_zorder(-np.inf)
        self.set_in_layout(False)

    def draw(self, renderer):
        ax = self.target()
        if ax is None:
            return
        short_side = min(ax.bbox.width, ax.bbox.height)
        axes = (ax.xaxis, ax.yaxis, ax.zaxis)
        font_px = renderer.points_to_pixels(max(a.label.get_fontsize() for a in axes))
        reserve = self.gap_fraction + font_px / max(short_side, 1.)
        ax.set_box_aspect(self.aspect, zoom=np.clip(1-2*reserve, .5, 1.))
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        for j, axis in enumerate(axes):
            start = limits[:,0].copy(); end = start.copy(); end[j] = limits[j,1]
            projected = np.array([proj3d.proj_transform(*p, ax.get_proj())[:2] for p in [start,end]])
            screen = ax.transData.transform(projected)
            length = np.linalg.norm(screen[1]-screen[0])
            widths = [text.get_window_extent(renderer).width for text in axis.get_ticklabels()]
            spacing = max(widths, default=font_px) + font_px
            locator = MaxNLocator(nbins=max(3, int(length/max(spacing,1.))))
            low, high = sorted(limits[j])
            ticks = locator.tick_values(low,high)
            edge_margin = 2*self.gap_fraction*(high-low)
            ticks = ticks[(ticks >= low+edge_margin) & (ticks <= high-edge_margin)]
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_tick_params(pad=self.gap_fraction*short_side/renderer.points_to_pixels(1.)/3)
            axis.label.set_visible(False)
        self.stale = False


class _Relative3DLabels(Artist):
    """Draw labels outside the projected grid and tick-text envelopes."""

    def __init__(self, ax, gap_fraction):
        super().__init__()
        self.target = weakref.ref(ax)
        self.gap_fraction = gap_fraction
        self.labels = [Text(ha='center', va='center', transform=IdentityTransform(),
                            clip_on=False, rotation_mode='anchor') for _ in range(3)]
        for text in self.labels:
            text.set_figure(ax.figure)
        self.set_zorder(np.inf)

    def draw(self, renderer):
        ax = self.target()
        if ax is None:
            return
        corners = np.array(list(itertools.product(ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())))
        projected = np.array([proj3d.proj_transform(*point,ax.get_proj())[:2] for point in corners])
        grid = ax.transData.transform(projected)
        center = grid.mean(axis=0)
        gap = self.gap_fraction * min(ax.bbox.width,ax.bbox.height)
        for axis, label in zip((ax.xaxis,ax.yaxis,ax.zaxis),self.labels):
            # The active projected edge follows camera changes automatically.
            line = ax.transData.transform(axis.line.get_xydata())
            midpoint = line.mean(axis=0)
            tangent = line[-1]-line[0]
            normal = np.array([-tangent[1],tangent[0]])
            normal /= max(np.linalg.norm(normal), np.finfo(float).eps)
            if normal @ (midpoint-center) < 0:
                normal = -normal
            label.set_text(axis.label.get_text())
            label.set_fontproperties(axis.label.get_fontproperties())
            label.set_color(axis.label.get_color())
            label.set_rotation(axis.label.get_rotation())
            label.set_position(midpoint)
            bbox = label.get_window_extent(renderer)
            half_extent = (abs(normal[0])*bbox.width+abs(normal[1])*bbox.height)/2
            support = np.max(grid@normal)
            low, high = sorted(axis.get_view_interval())
            # get_ticklabels() resets projected 3D text positions; inspect the
            # already drawn tick artists directly, ignoring out-of-view ticks.
            for tick in axis.majorTicks[:len(axis.get_majorticklocs())]:
                text = tick.label1
                if low <= tick.get_loc() <= high and text.get_visible():
                    support = max(support,np.max(text.get_window_extent(renderer).corners()@normal))
            anchor = midpoint+normal*(support-normal@midpoint+gap+half_extent)
            label.set_position(anchor)
            label.draw(renderer)
        self.stale = False

    def get_window_extent(self, renderer=None):
        return Bbox.union([text.get_window_extent(renderer) for text in self.labels])


def style_3d_axes(ax, label_gap_fraction=.04):
    """Place labels outside the grid with a gap relative to the axes area.

    Call after setting box aspect. Figure artists recompute spacing before and
    after the 3D axes render, including resize, camera rotation and PNG export.
    Offsets use the projected grid, rendered tick/text sizes and a dimensionless
    plot-area fraction; no fixed point/pixel label positions are used.
    """
    if not 0 < label_gap_fraction < .25:
        raise ValueError('Label gap fraction must lie between zero and .25')
    previous = getattr(ax,'_halo_relative_layout',None)
    if previous:
        for artist in previous:
            artist.remove()
    before = _Relative3DLayout(ax,label_gap_fraction)
    after = _Relative3DLabels(ax,label_gap_fraction)
    ax.figure.add_artist(before); ax.figure.add_artist(after)
    ax._halo_relative_layout = (before,after)
    return ax
