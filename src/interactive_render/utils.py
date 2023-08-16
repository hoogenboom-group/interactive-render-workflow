from tqdm.notebook import tqdm
import numpy as np
from skimage.exposure import rescale_intensity
import renderapi


def rescale_image(image, k=3):
    """Rescale intensity values.

    Clips intensity (based on mean +/- k*std) to facilitate feature finding, and
    decreases bit depth from 16bit to 8bit to save on memory.

    Parameters
    ----------
    image : (M, N) uint16 array
    k : scalar (optional)
        number of standard deviations away from mean from which to clip intensity
        defaults to 2

    Returns
    -------
    image : (M, N) ubyte array
    """
    mask = image[(image > 0) & (image < 65535)]
    vmin = int(mask.mean() - k*mask.std())
    vmax = int(mask.mean() + k*mask.std())
    return rescale_intensity(
        image,
        in_range=(vmin, vmax),
        out_range=np.ubyte
    )


def get_global_stack_bounds(stacks, **render):
    """Get global stack boundaries across multiple stacks"""

    # get all stack boundaries
    bounds_ = np.array(
        [list(renderapi.stack.get_stack_bounds(
            stack=stack,
            **render).values()) for stack in stacks]
    )
    # set global bounds
    bounds = {}
    bounds["minX"] = int(bounds_[:, 0].min())
    bounds["minY"] = int(bounds_[:, 1].min())
    bounds["minZ"] = int(bounds_[:, 2].min())
    bounds["maxX"] = int(bounds_[:, 3].max())
    bounds["maxY"] = int(bounds_[:, 4].max())
    bounds["maxZ"] = int(bounds_[:, 5].max())

    # set z values based on global z range
    z_values = list(range(bounds["minZ"], bounds["maxZ"] + 1))

    return bounds, z_values


def get_image_stacks(stacks, width=1000, **render):
    """Get image stacks"""
    # get global bounds and z values
    bounds, z_values = get_global_stack_bounds(stacks, **render)

    # initialize collection of images
    images = {stack: {} for stack in stacks}
    # loop through each stack
    for stack in tqdm(stacks):

        # render full section image at each z value
        for z in z_values:

            # render bbox image
            image = renderapi.image.get_bb_image(
                stack=stack,
                z=z,
                x=bounds['minX'],
                y=bounds['minY'],
                width=(bounds['maxX'] - bounds['minX']),
                height=(bounds['maxY'] - bounds['minY']),
                scale=(width / (bounds['maxX'] - bounds['minX'])),
                img_format='tiff16',
                **render
            )
            # add to collection
            images[stack][z] = image

    return images


def get_mosaic(stack, z, width=256, **render):
    """Get mosaic

    Most useful for plotting stitch lines. Differs from `get_image_stacks` in 
    that `get_mosaic` renders each tile in a grid with 0 spacing while 
    `get_image_stacks` renders bbox images of the entire section.

    Parameters
    ----------
    stack : str
    z : scalar
    width : scalar
        width of each tile in mosaic
    render : dict
    """
    # alias for width
    w = width

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

    # initialize big mosaic of the full image grid
    mosaic = np.zeros((w * grid_shape[0], w * grid_shape[1]))

    # loop through grid
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
        # fill in mosaic
        i = ts.layout.imageRow
        j = ts.layout.imageCol
        y1, y2 = i*w, (i + 1)*w
        x1, x2 = j*w, (j + 1)*w
        mosaic[y1:y2, x1:x2] = image

    return mosaic


def get_intrasection_pointmatches(
    stack,
    match_collection,
    **render
):
    """Get all intra-section point matches for a given stack

    Parameters
    ----------
    stack : str
    match_collection : str
    render : dict

    Returns
    -------
    d_matches : dict
        Mapping of ...
    """
    # Collection
    d_matches = {}

    # Get z values
    z_values = renderapi.stack.get_z_values_for_stack(
        stack=stack,
        **render
    )

    # Loop through each section
    for z in z_values:

        # Get sectionId from z value
        sectionId = renderapi.stack.get_sectionId_for_z(
            stack=stack,
            z=z,
            **render
        )

        # Get all the point matches within each section
        matches = renderapi.pointmatch.get_matches_with_group(
            pgroup=sectionId,
            matchCollection=match_collection,
            **render
        )

        # Add to collection
        d_matches[z] = matches

    return d_matches
