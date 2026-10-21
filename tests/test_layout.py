"""Check plot-relative 3D label clearance."""
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import proj3d
from halo_tracking.layout import style_3d_axes
from halo_tracking.study import load_config


def separated(a, b):
    """Convex-polygon separating-axis test in display coordinates."""
    for poly in [a,b]:
        for edge in np.roll(poly,-1,axis=0)-poly:
            normal=np.array([-edge[1],edge[0]])
            if (a@normal).max() <= (b@normal).min() or (b@normal).max() <= (a@normal).min():
                return True
    return False


def test_3d_labels_clear_grid_after_resize_rotation_and_export(tmp_path):
    fig=plt.figure(figsize=(8,6)); ax=fig.add_subplot(projection='3d')
    ax.plot([0,1],[0,1],[0,2]); ax.set(xlabel='Position x (km)',ylabel='Position y (km)',zlabel='Position z (km)')
    style_3d_axes(ax)
    for size, camera, dpi in [((8,6),(25,-60),100),((6,5),(35,35),150),((12,9),(15,120),100)]:
        fig.set_size_inches(*size); fig.set_dpi(dpi); ax.view_init(*camera); fig.canvas.draw()
        points=np.array(list(itertools.product(ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d())))
        projected=np.array([proj3d.proj_transform(*p,ax.get_proj())[:2] for p in points])
        grid=ax.transData.transform(projected); grid=grid[ConvexHull(grid).vertices]
        for text in ax._halo_relative_layout[1].labels:
            corners=text.get_window_extent(fig.canvas.get_renderer()).corners()[[0,1,3,2]]
            assert separated(corners,grid)
    fig.savefig(tmp_path/'layout.png',bbox_inches='tight')
    assert (tmp_path/'layout.png').stat().st_size>0
    plt.close(fig)

