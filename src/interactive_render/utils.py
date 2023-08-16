from tqdm.notebook import tqdm
import numpy as np
import renderapi


def get_global_stack_bounds(stacks, **render):
    """Get global stack boundaries across multiple stacks"""

    # Get all stack boundaries
    bounds_ = np.array(
        [list(renderapi.stack.get_stack_bounds(
            stack=stack,
            **render).values()) for stack in stacks]
    )
    # Set global bounds
    bounds = {}
    bounds["minX"] = int(bounds_[:, 0].min())
    bounds["minY"] = int(bounds_[:, 1].min())
    bounds["minZ"] = int(bounds_[:, 2].min())
    bounds["maxX"] = int(bounds_[:, 3].max())
    bounds["maxY"] = int(bounds_[:, 4].max())
    bounds["maxZ"] = int(bounds_[:, 5].max())

    # Set z values based on global z range
    z_values = list(range(bounds["minZ"], bounds["maxZ"] + 1))

    return bounds, z_values


def get_image_stacks(stacks, width=1000, **render):
    """Get image stacks"""
    # Get global bounds and z values
    bounds, z_values = get_global_stack_bounds(stacks, **render)

    # Initialize collection of images
    images = {stack: {} for stack in stacks}
    # Loop through each stack
    for stack in tqdm(stacks):

        # Render full section image at each z value
        for z in z_values:

            # Render bbox image
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
            # Add to collection
            images[stack][z] = image

    return images


def get_intrasection_pointmatches(
    stack,
    match_collection,
    **render
):
    """Get all intra-section point matches for a given stack

    Parameters
    ----------
    stack : str
        Input stack
    match_collection : str
        Pointmatch collection
    **render
        Keyword arguments for render-ws environment

    Returns
    -------
    d_matches : dict
        Mapping of 
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
