from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from ipywidgets import interact, fixed, IntSlider

import renderapi

from utils import (
    get_global_stack_bounds,
    get_image_stacks,
    get_intrasection_pointmatches
)


def f_plot_stacks(
    stacks,
    z,
    images,
    vmin=0,
    vmax=65535,
):
    """Support interactive plotting of multiple stacks"""

    # Create figure
    ncols = len(stacks)
    fig, axes = plt.subplots(
        ncols=ncols,
        sharex=True,
        sharey=True,
        figsize=(5*ncols, 5),
        squeeze=False,
    )
    # Map each stack to an axis
    axmap = {k: v for k, v in zip(stacks, axes.flat)}

    # Loop through stacks
    for stack in stacks:

        # Plot image
        axmap[stack].imshow(
            images[stack][z],
            cmap="Greys_r",
            vmin=vmin,
            vmax=vmax
        )
        # Aesthetics
        axmap[stack].set_title(stack)


def plot_stacks(
    stacks,
    render,
    width=1000,
    vmin=0,
    vmax=65535,
):
    """Plot stacks interactively"""

    # Get z values (bounds not needed)
    _, z_values = get_global_stack_bounds(stacks, **render)

    # Get image stacks
    images = get_image_stacks(stacks, width, **render)

    # Standard-deviation-based clipping (ignore 0 intensity pixels)
    k = 3
    imstack = np.stack([list(d.values()) for d in images.values()])
    vmin = imstack[imstack > 0].mean() - k*imstack[imstack > 0].std()
    vmax = imstack[imstack > 0].mean() + k*imstack[imstack > 0].std()

    # Interaction magic
    interact(
        f_plot_stacks,
        stacks=fixed(stacks),
        z=IntSlider(min=min(z_values), max=max(z_values)),
        images=fixed(images),
        width=fixed(width),
        vmin=IntSlider(vmin, 0, 65535),
        vmax=IntSlider(vmax, 0, 65535)
    )


def plot_stitching_matches_columnwise(
    stack,
    match_collection,
    z_values=None,
    width=256,
    **render
):
    """Make column-wise plot of pointmatches to show how successful stitching was

    Parameters
    ----------
    stack : str
        Input stack
    match_collection : str
        Collection of pointmatches
    z_values : list-like (optional)
        List of z values
    width : scalar (optional)
        Width (in pixels) of each tiny image in the mosaic
    render
        Keyword arguments for render-ws environment
    """
    # alias for width
    w = width

    # handle z values
    if z_values is None:
        z_values = renderapi.stack.get_z_values_for_stack(
            stack=stack,
            **render
        )

    # get intra-section point matches
    d_matches = get_intrasection_pointmatches(
        stack=stack,
        match_collection=match_collection,
        **render
    )

    # create figure
    ncols = len(z_values)
    fig, axes = plt.subplots(
        ncols=ncols,
        figsize=(6*ncols, 5),
        squeeze=False
    )
    axmap = {k: v for k, v in zip(z_values, axes.flat)}

    # loop through sections
    for z in tqdm(z_values):

        # infer shape of tile grid
        tilespecs = renderapi.tilespec.get_tile_specs_from_z(
            stack=stack,
            z=z,
            **render
        )
        grid_shape = (
            1 + max([ts.layout.imageRow for ts in tilespecs]),
            1 + max([ts.layout.imageCol for ts in tilespecs])
        )

        # Initialize big mosaic of the full image grid
        mosaic = np.zeros((w * grid_shape[0], w * grid_shape[1]))

        # Loop through grid
        for ts in tilespecs:

            # Get low-res image of each tile
            image = renderapi.image.get_tile_image_data(
                stack=stack,
                tileId=ts.tileId,
                scale=w/ts.width,
                excludeAllTransforms=True,
                img_format='tiff16',
                **render
            )
            # Fill in mosaic
            i = ts.layout.imageRow
            j = ts.layout.imageCol
            y1, y2 = i*w, (i + 1)*w
            x1, x2 = j*w, (j + 1)*w
            mosaic[y1:y2, x1:x2] = image

        # Plot mosaic
        vmin = np.mean([ts.minint for ts in tilespecs])
        vmax = np.mean([ts.maxint for ts in tilespecs])
        axmap[z].imshow(mosaic, cmap="Greys_r", vmin=vmin, vmax=vmax)

        # Plot pointmatches as a collection of line segments
        for d in d_matches[z]:

            # Get tile specifications for tile pair
            ts_p = renderapi.tilespec.get_tile_spec(stack=stack, tile=d["pId"], **render)
            ts_q = renderapi.tilespec.get_tile_spec(stack=stack, tile=d["qId"], **render)

            if (ts_p is not None) and (ts_q is not None):
                # Get pointmatches for tile p, scale and shift them over
                i_p = ts_p.layout.imageRow
                j_p = ts_p.layout.imageCol
                X_p = np.array(d["matches"]["p"][0]) * (w/ts.width) + j_p*w
                Y_p = np.array(d["matches"]["p"][1]) * (w/ts.width) + i_p*w

                # Get pointmatches for tile q, scale and shift them over
                i_q = ts_q.layout.imageRow
                j_q = ts_q.layout.imageCol
                X_q = np.array(d["matches"]["q"][0]) * (w/ts.width) + j_q*w
                Y_q = np.array(d["matches"]["q"][1]) * (w/ts.width) + i_q*w

                # Convert pointmatch coordinates into line segments
                vertices = [[(x_p, y_p), (x_q, y_q)] for (x_p, y_p, x_q, y_q) in zip(X_p, Y_p, X_q, Y_q)]
                lines = LineCollection(vertices, color="#ffaa00")
                axmap[z].add_collection(lines)

        # Aesthetics
        title = f"Z = {z}"
        axmap[z].set_title(title)
