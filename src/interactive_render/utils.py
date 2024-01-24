import requests
from tqdm.notebook import tqdm
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.transform import pyramid_gaussian
import renderapi
from scripted_render_pipeline.importer.render_specs import Axis, Tile, Section, Stack
from scripted_render_pipeline.importer import fastem_mipmapper
from renderapi.errors import RenderError
import pathlib
import tifffile

SCALE = 0.05 # Scale at which to render downsampled images
BASE_URL = ""  # "file://"

def clear_image_cache():
    url = "https://sonic.tnw.tudelft.nl/render-ws/v1/imageProcessorCache/allEntries"
    response  = requests.delete(url)
    if response.status_code == 401:  # probably running locally
        url = "http://localhost:8081/render-ws/v1/imageProcessorCache/allEntries"
        response = requests.delete(url)
    return response


def rescale_image(image, k=3, out_range=np.uint16):
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
        out_range=out_range
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
                maxTileSpecsToRender=10000,
                img_format='tiff16',
                **render
            )
            if not isinstance(image, np.ndarray): # If somehow the get_bb_image failed...
                image = np.zeros(images[stack][z-1].shape)
            # add to collection
            images[stack][z] = image
    return images

def create_downsampled_stack(project_dir, stack_2_downsample, trakem=False, **render):
    """Create downsampled image stack

    Parameters
    ----------
    project_dir : Path
        directory where project data is stored
    stack_2_downsample : dict 
        input and output stack name
    trakem: Bool
        whether to make stack for importing into TrakEM2 to do manual rough alignment
    render : dict
        render parameters

    returns "ds_stack" object
    """
    all_sections = {}
    stack = stack_2_downsample['in']
    if trakem:
        ds_stack_name = f"{stack_2_downsample['out']}_trakem"
    else:
        ds_stack_name = stack_2_downsample['out']
    bounds = renderapi.stack.get_stack_bounds(stack, 
                                              **render)
    z_values = renderapi.stack.get_z_values_for_stack(stack, 
                                                      **render)                                                   
    ds_stack_dir = project_dir / "_dsmontages" / ds_stack_name
    ds_stack_dir.mkdir(parents=True, exist_ok=True)
    
    # Loop through z levels
    for z in tqdm(z_values, 
                  total=len(z_values)):
        sectionId = renderapi.stack.get_sectionId_for_z(stack, 
                                                        z=z,
                                                        **render)
        # Get downsampled image of section
        ds_image = renderapi.image.get_bb_image(stack=stack,
                                               z=z,
                                               x=bounds['minX'],
                                               y=bounds['minY'],
                                               width=(bounds['maxX'] - bounds['minX']),
                                               height=(bounds['maxY'] - bounds['minY']),
                                               scale=SCALE,
                                               maxTileSpecsToRender=10000,
                                               img_format='tiff16',
                                               **render)
        ds_image = rescale_image(ds_image, out_range=np.uint8) # Rescale intensities and to 8-bit
        
        # Make imagePyramid
        if trakem: # imagePyramid with one level
            leveldict = {}
            new_file_name = f"{sectionId}.tif"
            new_file_path = ds_stack_dir / new_file_name
            with tifffile.TiffWriter(new_file_path) as fp:
                fp.write(ds_image.astype(np.uint8))
            url = BASE_URL + new_file_path.as_posix()
            leveldict[0] = renderapi.image_pyramid.MipMap(url) # One level...
            pyramid = renderapi.image_pyramid.ImagePyramid(leveldict)
        else: # Actually make a pyramid
            ds_section_dir = ds_stack_dir / sectionId
            ds_section_dir.mkdir(parents=True, exist_ok=True)
            mipmapper = fastem_mipmapper.FASTEM_Mipmapper(project_path=project_dir)
            pyramid = mipmapper.make_pyramid(ds_section_dir,
                                             ds_image,
                                             description=None)
        # TileSpecs for building render stack
        layout = renderapi.tilespec.Layout(
            sectionId=sectionId,
            imageRow=0,
            imageCol=0
        )
        width, height = ds_image.shape[1], ds_image.shape[0]
        spec = renderapi.tilespec.TileSpec(
            imagePyramid=pyramid,
            width=width,
            height=height,
            layout=layout,
            tforms=[]
        )
        pixels = width, height
        mins = [min(0, value) for value in pixels]
        maxs = [max(0, value) for value in pixels]
        coordinates = [xy * px for xy, px in zip([0, 0], pixels)]
        axes = [Axis(*item) for item in zip(mins, maxs, coordinates)]
        # Generate Tile and Section instances
        ds_tile = Tile(stackname=ds_stack_name, zvalue=z, spec=spec, 
                       acquisitiontime=None, axes=axes, 
                       min_intensity=0, max_intensity=255)
        section = Section(ds_tile.zvalue, ds_tile.stackname)
        # Add Tile to 'Section' (Tile = section image)
        section.add_tile(ds_tile)
        all_sections[ds_tile.zvalue] = section
  
    # Add sections (ds_tiles) to stack
    ds_stack = Stack(ds_stack_name)
    for section in all_sections.values():
        ds_stack.add_section(section)

    return ds_stack # Stack of downsampled images

    
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

    # Compute grid shape from stack bounds and max tile width and height
    metadata = renderapi.stack.get_full_stack_metadata(
        stack=stack,
        **render
    )
    bounds = metadata['stats']['stackBounds']
    grid_shape = (
        int(bounds['maxY'] / metadata['stats']['maxTileHeight']),
        int(bounds['maxX'] / metadata['stats']['maxTileWidth'])
    )

    # initialize big mosaic of the full image grid
    mosaic = np.zeros((w * grid_shape[0], w * grid_shape[1]))
    
    # get tilespecs
    tilespecs = renderapi.tilespec.get_tile_specs_from_z(
        stack=stack,
        z=z,
        **render
        )

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


def get_stitching_pointmatches(
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

        # Get all the point matches 
        matches = renderapi.pointmatch.get_matches_within_group(
            groupId=sectionId,
            matchCollection=match_collection,
            **render
        )

        # Add to collection
        d_matches[z] = matches

    return d_matches

def get_alignment_pointmatches(
    stack,
    match_collection,
    **render
):
    """Get all inter-section point matches for a given stack
    Only retrieves matches between sections adjacent in z
    Since its difficult to plot matches further away

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
    for z in z_values[:-1]:

        # Get sectionId from z value
        psectionId = renderapi.stack.get_sectionId_for_z(
            stack=stack,
            z=z,
            **render
        )
        
        # Get sectionId from z+1 value
        qsectionId = renderapi.stack.get_sectionId_for_z(
            stack=stack,
            z=z+1,
            **render
        )

        # Get all the point matches 
        matches = renderapi.pointmatch.get_matches_from_group_to_group(
            pgroup=psectionId,
            qgroup=qsectionId,
            matchCollection=match_collection,
            **render
        )
        
        # Add to collection
        d_matches[z] = matches

    return d_matches

def update_stack_resolution(
        stack,
        stackresolutionX,
        stackresolutionY,
        stackresolutionZ,
        **render
):
    """Update the stack resolution
    
    Parameters
    ----------
    stack : str
    stackresolutionX : float, resolution in X
    stackresolutionY : float, resolution in Y
    stackresolutionZ : float, resolution in Z
    """
    # Get stack metadata
    stackversion = renderapi.stack.get_stack_metadata(stack,
                                                      **render)
    stackversion.stackResolutionX = stackresolutionX
    stackversion.stackResolutionY = stackresolutionY
    stackversion.stackResolutionZ = stackresolutionZ
    # Update metadata
    renderapi.stack.set_stack_metadata(stack,
                                       stackversion,
                                       **render)
    renderapi.stack.set_stack_state(stack,
                                    state='COMPLETE',
                                    **render)

def render_aligned_tiles(
        stack, 
        width=1024,
        **render):
    """Renders images of all z levels in the aligned stack

    Parameters
    ----------
    stack : str
        Stack from which to render neighborhood image
    width : float
        Width of rendered layer images in pixels
    render : `renderapi.render.RenderClient`
        `render-ws` instance
    """
    images = {}
    bounds = renderapi.stack.get_stack_bounds(stack=stack,
                                              **render)
    z_values = renderapi.stack.get_z_values_for_stack(stack=stack,
                                                      **render)
    # Get bbox of center of stack with size specified by input width
    centerX = bounds['minX'] + bounds['maxX'] / 2
    centerY = bounds['minY'] + bounds['maxY'] / 2
    bbox = (int(centerX - 0.5 * width),
            int(centerY - 0.5 * width),
            width,
            width)
    # Loop over z values in stack
    for z in z_values:
        # Render bbox
        image = renderapi.image.get_bb_image(stack=stack,
                                             z=z,
                                             x=bbox[0],
                                             y=bbox[1],
                                             width=bbox[2],
                                             height=bbox[3],
                                             scale=0.5,
                                             **render)
        images[z] = image[:,:,0]
    return images 