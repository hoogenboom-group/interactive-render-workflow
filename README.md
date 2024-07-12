[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12733815.svg)](https://doi.org/10.5281/zenodo.12733815)
# interactive-render-workflow
Interactive workflow for aligning volume electron microscopy datasets derived from [FAST-EM](https://www.delmic.com/en/products/fast-imaging/fast-em), making use of [render-ws](https://github.com/saalfeldlab/render). This software is being used to reconstruct FAST-EM array tomography datasets as described in [Kievits et al. (2024)](https://doi.org/10.1515/mim-2024-0005). The software performs interactive post-correction of FAST-EM image data, import to `render-ws`, 2D stitching, 3D alignment and export to WebKnossos. 

The following is currently supported:
- Post-correction of FAST-EM image data
- Import to `render-ws`
- 2D stitching
- 3D alignment
- Export to local WebKnossos instances

## Requirements
- Server with Linux distribution (Ubutuntu) and decent computation power (>128 GB RAM, >40 CPU cores).
- [render-ws](https://github.com/saalfeldlab/render/blob/b06be441f3c78e1423c54bce20b291752c6d0773/docs/src/site/markdown/render-ws.md) installation ([setup instructions](https://github.com/hoogenboom-group/em-infrastructure/blob/master/docs/Render-ws.md))
- Local WebKnossos instance ([setup instructions](https://github.com/hoogenboom-group/em-infrastructure/blob/master/docs/Webknossos.md)). Since we are using a self-hosted WebKnossos instance, export to the [Remote WebKnossos](https://webknossos.org/) is currently not supported be can be considered on request. 
- [scripted-render-pipeline](https://github.com/hoogenboom-group/scripted-render-pipeline) installation

## Installation
It is recommended to install `interactive-render-workflow` in a [Python virtual environment](https://docs.python.org/3/library/venv.html) or with help of a Python environment manager such as [Conda](https://docs.conda.io/en/latest/) or [Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html), to prevent changes to your system Python installation.

Instructions for venv (Python virtual environment, requires Python3.10 installation)
```
python3 -m venv /path/to/new/virtual/environment
```
Activate environment:
```
source /path/to/new/virtual/environment/bin/activate
```
Install repository
```
pip install git+https://github.com/hoogenboom-group/interactive-render-workflow.git
```
Clone `BigFeta` into a suitable directory (required for alignment scripts)
```
git clone git://github.com/AllenInstitute/BigFeta/
```

## Usage
The usage is covered entirely by JuPyter Notebooks:
1. `post-correct.ipynb` is for optimizing post-correction for a single section and then applying these settings to all sections
2. `import.ipynb` is for import to `render-ws`
3. `stitch_render-client.ipynb` uses the `render-ws` routines to perform in-plane (2D) alignment (stitching)
4. `rough_align_render-client.ipynb` performs rough alignment (as input to the final fine alignemnt step)
5. `fine_align_render-client.ipynb` performs the fine alignment
6. `export.ipynb` performs export of 3D aligned stacks to WebKnossos

We are currently writing routines to perform stitching and alignment using alternative feature-finding algorithm implementations (`stitch.ipynb`, `align.ipynb`). These are still under construction.

## License
Licensed under the GNU Public license, Version 3.0 (the "License"); you may not use this software except in compliance with the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Support
This code is an important part of the Hoogenboom group code base and we are actively using and maintaining it. This means that the documentation and API may be subject to changes. Issues are encouraged, but this software is released with no fixed update schedule.
