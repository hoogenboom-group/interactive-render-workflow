import numpy as np
from skimage.exposure import rescale_intensity
import renderapi
from renderapi.errors import RenderError


OVERLAP = 0.05  # assumed overlap between fields
BUFFER = 100    # amount of extra pixels to include in bbox


def get_bbox_from_relative_position(
    tilespec,
    relative_position,
):
    """Determine bounding box from relative position.

    Parameters
    ----------
    tilespec : `renderapi.tilespec.TileSpec`
    relative_position : str
        relative position as determined by `renderapi.client.tilePairClient`

    Returns
    -------
    bbox : 4-tuple
        (x_min, y_min, x_max, y_max)
    """
    if relative_position is None:
        return tilespec.bbox

    # unpack bounding box
    x_min, y_min, x_max, y_max = tilespec.bbox

    # change the appropriate coordinate
    if relative_position.lower() == "left":
        x_min = x_max - OVERLAP*tilespec.width - BUFFER
    elif relative_position.lower() == "right":
        x_max = x_min + OVERLAP*tilespec.width + BUFFER
    elif relative_position.lower() == "top":
        y_min = y_max - OVERLAP*tilespec.height - BUFFER
    elif relative_position.lower() == "bottom":
        y_max = y_min + OVERLAP*tilespec.height + BUFFER

    else:
        msg = f"unknown relative position, '{relative_position}'."
        raise ValueError(msg)

    bbox = (x_min, y_min, x_max, y_max)
    return bbox


def rescale_image_for_matching(image, k=2):
    """Rescale intensity values prior to feature finding.

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


def get_image_for_matching(
    stack,
    tileId,
    relative_position=None,
    **render_kwargs
):
    """Retrieve (part of) an image expected to overlap with its pair.

    Parameters
    ----------
    stack : str
    tileId : str
    relative_position : str (optional)
        relative position as determined by `renderapi.client.tilePairClient`
        defaults to None --> full image

    Returns
    -------
    image : (M, N) ubyte array
    """
    # get tile specification to determine bbox
    spec = renderapi.tilespec.get_tile_spec(
        stack=stack,
        tile=tileId,
        **render_kwargs
    )
    if spec is None:
        msg = f"tile spec, '{tileId}', does not exist in stack, '{stack}'."
        return RenderError(msg)

    # determine bbox from relative position and overlap
    bbox = get_bbox_from_relative_position(
        spec, relative_position
    )

    # get image as 16bit tiff
    # TODO: why `maxTileSpecsToRender` fails at low values?
    image = renderapi.image.get_bb_image(
        stack,
        z=spec.z,
        x=bbox[0],
        y=bbox[1],
        width=bbox[2] - bbox[0],
        height=bbox[3] - bbox[1],
        scale=1.0,
        maxTileSpecsToRender=32,
        img_format="tiff16",
        **render_kwargs
    )
    if isinstance(image, RenderError):
        return image

    return rescale_image_for_matching(image)


def get_image_pair_for_matching(
    stack,
    tilepair,
    **render_kwargs
):
    """Get pair of images from which to find features.

    Parameters
    ----------
    stack : str
    tilepair : dict
        output element from `renderapi.client.tilePairClient`
        {"p": {"groupId": ...,
               "id": ...,
               "relativePosition": ...},
         "q": {"groupId": ...,
               "id": ...,
               "relativePosition": ...}
        }

    Returns
    -------
    image_p : (M, N) ubyte array
    image_q : (M, N) ubyte array
    """
    image_p = get_image_for_matching(
        stack=stack,
        tileId=tilepair["p"].get("id"),
        relative_position=tilepair["p"].get("relativePosition"),
        **render_kwargs
    )
    image_q = get_image_for_matching(
        stack=stack,
        tileId=tilepair["q"].get("id"),
        relative_position=tilepair["q"].get("relativePosition"),
        **render_kwargs
    )
    return image_p, image_q
