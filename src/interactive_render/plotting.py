import numpy as np
import matplotlib.pyplot as plt
import renderapi
import tifffile

from matplotlib.collections import LineCollection
from ipywidgets import interact, fixed, IntSlider, FloatSlider, FloatText
from tqdm.notebook import tqdm
from scripted_render_pipeline.postcorrector.post_corrector import Post_Corrector

from .utils import (
    rescale_image,
    get_global_stack_bounds,
    get_image_stacks,
    get_mosaic,
    get_stitching_pointmatches,
    get_alignment_pointmatches,
    render_aligned_tiles
)

WIDTH_FIELD = 6400


def f_plot_stacks(
    stacks,
    z,
    images,
    vmin=0,
    vmax=65535,
):
    """Support interactive plotting of multiple stacks"""

    # create figure
    ncols = len(stacks)
    fig, axes = plt.subplots(
        ncols=ncols,
        sharex=True,
        sharey=True,
        figsize=(5*ncols, 5),
        squeeze=False,
    )
    # map each stack to an axis
    axmap = {k: v for k, v in zip(stacks, axes.flat)}

    # loop through stacks
    for stack in stacks:

        # plot image
        axmap[stack].imshow(
            images[stack][z],
            cmap="Greys_r",
            vmin=vmin,
            vmax=vmax
        )
        # aesthetics
        axmap[stack].set_title(stack)


def plot_stacks(
    stacks,
    width=1000,
    vmin=0,
    vmax=65535,
    **render
):
    """Plot stacks interactively"""

    # get z values (bounds not needed)
    _, z_values = get_global_stack_bounds(stacks, **render)

    # get image stacks
    images = get_image_stacks(stacks, width, **render)

    # standard-deviation-based clipping (ignore 0 intensity pixels)
    k = 3
    imstack = np.stack([list(d.values()) for d in images.values()])
    vmin = imstack[imstack > 0].mean() - k*imstack[imstack > 0].std()
    vmax = imstack[imstack > 0].mean() + k*imstack[imstack > 0].std()

    # interaction magic
    interact(
        f_plot_stacks,
        stacks=fixed(stacks),
        z=IntSlider(min=min(z_values), max=max(z_values)),
        images=fixed(images),
        width=fixed(width),
        vmin=IntSlider(vmin, 0, 65535),
        vmax=IntSlider(vmax, 0, 65535)
    )


def f_plot_stack_with_matches(
    stack,
    z,
    mosaics,
    d_matches,
    width,
    render,
):
    """Support interactive plotting of a stack with overlaid stitch lines"""
    # alias for width
    w = width

    # create figure
    fig, ax = plt.subplots(figsize=(5, 5))

    # plot mosaic
    ax.imshow(
        mosaics[z],
        cmap="Greys_r",
    )

    # loop through tile-2-tile point matches for given section
    for d in d_matches[z]:

        # get tile specifications for tile pair
        ts_p = renderapi.tilespec.get_tile_spec(stack=stack, tile=d["pId"], **render)
        ts_q = renderapi.tilespec.get_tile_spec(stack=stack, tile=d["qId"], **render)

        if (ts_p is not None) and (ts_q is not None):

            # get pointmatches for tile p, scale and shift them over
            i_p = ts_p.layout.imageRow
            j_p = ts_p.layout.imageCol
            X_p = np.array(d["matches"]["p"][0]) * (w/WIDTH_FIELD) + j_p*w
            Y_p = np.array(d["matches"]["p"][1]) * (w/WIDTH_FIELD) + i_p*w
                
            # get pointmatches for tile q, scale and shift them over
            i_q = ts_q.layout.imageRow
            j_q = ts_q.layout.imageCol
            try:
                X_q = np.array(d["matches"]["q"][0]) * (w/WIDTH_FIELD) + j_q*w
                Y_q = np.array(d["matches"]["q"][1]) * (w/WIDTH_FIELD) + i_q*w
            except IndexError:
                X_q = np.array([])
                Y_q = np.array([])

            # convert pointmatch coordinates into line segments
            vertices = [[(x_p, y_p), (x_q, y_q)] for (x_p, y_p, x_q, y_q) in zip(X_p, Y_p, X_q, Y_q)]
            lines = LineCollection(vertices, color="#ffaa00")

            # plot stitch lines and annotate
            ax.add_collection(lines)
            x = max(j_p, j_q) * w if (i_p == i_q) else (j_p + 0.5) * w
            y = (i_p + 0.5) * w if (i_p == i_q) else max(i_p, i_q) * w
            s = f"{len(vertices)}"
            ax.text(x, y, s=s, ha="center", va="center",
                    bbox={"facecolor": "none", "edgecolor": "black", "pad": 2})


def f_plot_dsstack_with_matches(
    stack,
    z,
    dsimages,
    d_matches,
    width,
    render,
):
    """Support interactive plotting of a downsampled image stack with overlaid matches in z"""
    # width and height plot
    w = width
    h = dsimages[z].shape[0]

    # create figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # plot dsstack z-level
    ax[0].imshow(
        dsimages[z],
        cmap="Greys_r",
    )
    # plot dsstack adjacent z-level
    ax[1].imshow(
        dsimages[z+1],
        cmap="Greys_r",
    )
    # loop through point matches for given section-pair
    for d in d_matches[z]:
        if d:
            # get tile specifications for tile pair
            ts_p = renderapi.tilespec.get_tile_spec(stack=stack, tile=d["pId"], **render)
            ts_q = renderapi.tilespec.get_tile_spec(stack=stack, tile=d["qId"], **render)

            if (ts_p is not None) and (ts_q is not None):
                # get pointmatches for tile p, scale
                X_p = np.array(d["matches"]["p"][0]) * (w / ts_p.width) 
                Y_p = np.array(d["matches"]["p"][1]) * (w / ts_p.height) 

                # get pointmatches for tile q, scale
                try:
                    X_q = np.array(d["matches"]["q"][0]) * (w / ts_q.width) 
                    Y_q = np.array(d["matches"]["q"][1]) * (w / ts_q.height) 
                except IndexError:
                    X_q = np.array([])
                    Y_q = np.array([])
        else:
            X_q = np.array([])
            Y_q = np.array([])

        # Plot matches
        ax[0].scatter(X_p, Y_p, color='orange', marker='x')
        ax[1].scatter(X_q, Y_q, color='blue', marker='x')
        s_p, s_q = len(X_p), len(X_q)
        ax[0].text(h/2, w/2, s=s_p, ha="center", va="center",
                bbox={"facecolor": "none", "edgecolor": "black", "pad": 2})
        ax[1].text(h/2, w/2, s=s_q, ha="center", va="center",
                bbox={"facecolor": "none", "edgecolor": "black", "pad": 2})
        
        # Aesthetics
        ax[0].set_title(d['pGroupId'])
        ax[1].set_title(d['qGroupId'])


def plot_stack_with_stitching_matches(
    stack,
    match_collection,
    render,
    width=256
):
    """Plot stack interactively and overlay stitch lines"""
    # get z values (bounds not needed)
    _, z_values = get_global_stack_bounds([stack], **render)

    # get stack of mosaics
    mosaics = rescale_image(
        np.stack([get_mosaic(stack, z, width, **render) for z in z_values]),
        k=3,
    )

    # get intra-section point matches
    d_matches = get_stitching_pointmatches(
        stack,
        match_collection,
        **render
    )

    # interaction magic
    interact(
        f_plot_stack_with_matches,
        stack=fixed(stack),
        z=IntSlider(min=min(z_values), max=max(z_values)),
        mosaics=fixed(mosaics),
        d_matches=fixed(d_matches),
        width=fixed(width),
        render=fixed(render)
    )

def plot_dsstack_with_alignment_matches(
    stack,
    match_collection,
    width,
    **render
):
    """Plot stack interactively and show matches between z-levels"""
    # get z values (bounds not needed)
    _, z_values = get_global_stack_bounds([stack], **render)

    # get dsimage stack
    dsimages = get_image_stacks([stack],
                                width=width,
                                **render)

    # get inter-section point matches
    d_matches = get_alignment_pointmatches(
        stack,
        match_collection,
        **render
    )

    # interaction magic
    interact(
        f_plot_dsstack_with_matches,
        stack=fixed(stack),
        dsimages=fixed(dsimages[stack]),
        z=IntSlider(min=min(z_values), max=max(z_values)-1),
        d_matches=fixed(d_matches),
        width=fixed(width),
        render=fixed(render)
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
        input stack
    match_collection : str
        collection of pointmatches
    z_values : list-like (optional)
        list of z values
    width : scalar (optional)
        width (in pixels) of each tiny image in the mosaic
    render
        keyword arguments for render-ws environment
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
    d_matches = get_stitching_pointmatches(
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
        
        # get mosaic
        mosaic = rescale_image(
            get_mosaic(stack, z, width=w, **render)
        )
        # plot mosaic
        axmap[z].imshow(mosaic, cmap="Greys_r")

        # plot pointmatches as a collection of line segments
        for d in d_matches[z]:

            # get tile specifications for tile pair
            ts_p = renderapi.tilespec.get_tile_spec(stack=stack, tile=d["pId"], **render)
            ts_q = renderapi.tilespec.get_tile_spec(stack=stack, tile=d["qId"], **render)

            if (ts_p is not None) and (ts_q is not None):
                # get pointmatches for tile p, scale and shift them over
                i_p = ts_p.layout.imageRow
                j_p = ts_p.layout.imageCol
                X_p = np.array(d["matches"]["p"][0]) * (w/WIDTH_FIELD) + j_p*w
                Y_p = np.array(d["matches"]["p"][1]) * (w/WIDTH_FIELD) + i_p*w

                # get pointmatches for tile q, scale and shift them over
                i_q = ts_q.layout.imageRow
                j_q = ts_q.layout.imageCol
                X_q = np.array(d["matches"]["q"][0]) * (w/WIDTH_FIELD) + j_q*w
                Y_q = np.array(d["matches"]["q"][1]) * (w/WIDTH_FIELD) + i_q*w

                # convert pointmatch coordinates into line segments
                vertices = [[(x_p, y_p), (x_q, y_q)] for (x_p, y_p, x_q, y_q) in zip(X_p, Y_p, X_q, Y_q)]
                lines = LineCollection(vertices, color="#ffaa00")
                axmap[z].add_collection(lines)

        # aesthetics
        title = f"Z = {z}"
        axmap[z].set_title(title)


def f_plot_aligned_stack_with_matches(
    stack,
    z,
    mosaics,
    d_matches,
    width,
    render,
):
    """Support interactive plotting of a roughly aligned image stack with overlaid fine matches in z"""
    # width and height plot
    w = width

    # create figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # plot mosaic
    ax[0].imshow(
        mosaics[z],
        cmap="Greys_r",
    )

    ax[1].imshow(
        mosaics[z+1],
        cmap="Greys_r",
    )

    # loop through tilepairs in sections pair
    for d in d_matches[z]:

        # get tile specifications for tilepair
        ts_p = renderapi.tilespec.get_tile_spec(stack=stack, tile=d["pId"], **render)
        ts_q = renderapi.tilespec.get_tile_spec(stack=stack, tile=d["qId"], **render)

        if (ts_p is not None) and (ts_q is not None):
            
            # get pointmatches for tile p, scale and shift them over
            i_p = ts_p.layout.imageRow
            j_p = ts_p.layout.imageCol
            X_p = np.array(d["matches"]["p"][0]) * (w/WIDTH_FIELD) + j_p*w
            Y_p = np.array(d["matches"]["p"][1]) * (w/WIDTH_FIELD) + i_p*w

            # get pointmatches for tile q, scale and shift them over
            i_q = ts_q.layout.imageRow
            j_q = ts_q.layout.imageCol
            X_q = np.array(d["matches"]["q"][0]) * (w/WIDTH_FIELD) + j_q*w
            Y_q = np.array(d["matches"]["q"][1]) * (w/WIDTH_FIELD) + i_q*w

            # Plot matches
            ax[0].scatter(X_p, Y_p, color='orange', marker='x')
            ax[1].scatter(X_q, Y_q, color='blue', marker='x')
            s_p, s_q = len(X_p), len(X_q)
            ax[0].text(w/2, w/2, s=s_p, ha="center", va="center",
                    bbox={"facecolor": "none", "edgecolor": "black", "pad": 2})
            ax[1].text(w/2, w/2, s=s_q, ha="center", va="center",
                    bbox={"facecolor": "none", "edgecolor": "black", "pad": 2})
            
            # Aesthetics
            ax[0].set_title(d['pGroupId'])
            ax[1].set_title(d['qGroupId'])


def plot_aligned_stack_with_alignment_matches(
    stack,
    match_collection,
    width=256,
    **render
):
    """Plot rough aligned stack interactively and show matches between z-levels"""
    # get z values (bounds not needed)
    _, z_values = get_global_stack_bounds([stack], **render)

    # get stack of mosaics
    mosaics = rescale_image(
        np.stack([get_mosaic(stack, z, width, **render) for z in z_values]),
        k=3,
    )

    # get inter-section point matches
    d_matches = get_alignment_pointmatches(
        stack,
        match_collection,
        **render
    )

    # interaction magic
    interact(
        f_plot_aligned_stack_with_matches,
        stack=fixed(stack),
        z=IntSlider(min=min(z_values), max=max(z_values)-1),
        mosaics=fixed(mosaics),
        d_matches=fixed(d_matches),
        width=fixed(width),
        render=fixed(render)
    )

def plot_aligned_tiles(
    stack,
    width=1024,
    alpha=0.5,
    **render
):
    """Plot tile_pair in z and neighbors to show alignment"""
    # get bb images of tile_pair
    images = render_aligned_tiles(stack=stack,
                                  width=width,
                                  **render)
    z_values = renderapi.stack.get_z_values_for_stack(stack=stack,
                                                      **render)
    # interaction magic
    interact(
        f_plot_aligned_tiles,
        z=IntSlider(min=min(z_values), max=max(z_values)-1),
        images=fixed(images),
        alpha1=FloatSlider(value=alpha, min=0, max=1, step=0.01, description='alpha1'),
        alpha2=FloatSlider(value=alpha, min=0, max=1, step=0.01, description='alpha2')
    )

def f_plot_aligned_tiles(
    z,
    images,
    alpha1=0.5,
    alpha2=0.5
):
    """Support interactive plotting of an aligned stack"""
    # Extent
    extent = np.min(images[z]), np.max(images[z]), np.min(images[z]), np.max(images[z])
    # Create figure
    fig = plt.figure(figsize=(4,4))
    # Add images
    im1 = plt.imshow(images[z], cmap="Greens_r", alpha=alpha1, extent=extent)
    im2 = plt.imshow(images[z+1], cmap="Oranges_r", alpha=alpha2, extent=extent)
    
def plot_images_with_artefacts(
    filepaths,
    med,
    mad,
    pct=0.1,
    a=1,
    vmin=0,
    vmax=65535    
):
    """Plot images with artefacts for outlier detection parameter optimization"""
    # Get raw images
    raw_images = []
    for fp in tqdm(filepaths):
        tiff = tifffile.TiffFile(fp.as_posix()) # Read tiff and extract lowest resolution page from pyramid
        image = tiff.pages[-1].asarray()
        raw_images.append(image)
        
    # interaction magic
    interact(
        f_plot_images_with_artefacts,
        filepaths=fixed(filepaths),
        images=fixed(raw_images),
        med=fixed(med),
        mad=fixed(mad),
        pct=FloatText(value=pct, description='percentile', disabled=False),
        a=FloatText(value=a, description='scaling factor a', disabled=False),
        vmin=IntSlider(vmin, 0, 65535),
        vmax=IntSlider(vmax, 0, 65535), 
    )

def f_plot_images_with_artefacts(
    filepaths,
    images,
    med,
    mad,
    pct,
    a,
    vmin=0,
    vmax=65535      
):  
    """Support interactive plotting of images with artifacts"""
    postcorrector = Post_Corrector(filepaths)
    # Create figure
    nrows, ncols, _ = [int(i) + 1 for i in filepaths[-1].stem.split('_')]
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize=(2*ncols, 2*nrows))
    # Loop over raw image files
    for fp, im in zip(filepaths, images):
        # Determine row, col
        row, col, _ = [int(i) for i in fp.stem.split('_')]
        # Detect artefacts
        corrupted = postcorrector.has_artefact(im, med=med, mad=mad, pct=pct, a=a)
        # Colorize
        tint_green = np.array([0, 1, 0, 0.1])
        tint_red = np.array([1, 0, 0, 0.1])
        dy, dx = im.shape
        if corrupted:
            mask = np.ones((dy, dx, 4)) * tint_red
        else:
            mask = np.ones((dy, dx, 4)) * tint_green

        # Plot image + mask
        axes[row, col].imshow(rescale_image(im), cmap='Greys_r',
                              vmin=vmin, vmax=vmax)
        axes[row, col].imshow(mask)

        # Give label
        title = f"{row:03d} x {col:03d}"
        axes[row, col].text(0.5, 0.95, title, ha='center', va='top',
                            transform=axes[row, col].transAxes,
                            fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        # Remove axis ticks
        axes[row, col].axis('off')
    
    # Aesthetics
    plt.subplots_adjust(hspace=0.02, wspace=-0.02)
    
    
    