# Data Preprocessing

> This section of code is built on [CLAM](https://github.com/mahmoodlab/CLAM) by Mahmood Lab at Harvard Medical School, and [Aslide](https://github.com/MrPeterJin/ASlide) built by Cheng Jin.

## Enhanced Features

The modified pipeline includes several improvements:

- **JSON-based Input**: Reads slide paths from a JSON manifest instead of directly from a directory, allowing for more flexible data organization
- **Multicore Processing**: Supports parallel processing using multiple CPU cores

The following is the description of code usage and the process of data preprocessing.

## Step 1: Building the WSI JSON Manifest

Before processing slides, we need to create a JSON manifest file that lists all WSIs to be processed. The `create_patching_json.py` script handles this task.

### Usage

```bash
python scan_wsi.py /path/to/slides --recursive --output slides_manifest.json --extensions .svs .ndpi .tiff
```

### JSON Structure

The output JSON file should have the following structure:

```json
{
    "images": [
        {
            "id": 0,
            "original_path": "/path/to/slide1.svs"
        },
        {
            "id": 1,
            "original_path": "/path/to/slide2.svs"
        },
        ...
    ]
}
```

## Step 2: JSON-based WSI Processing Pipeline

For processing slides based on the JSON manifest, we use the `create_patches_fp.py` script. This script expects the following arguments:

- `source`: JSON file with paths to whole slide images (.svs, .ndpi, .tiff, etc.)
- `patch_size`: Size of square patches (e.g., 1200×1200 pixels in the modified version)
- `step_size`: Stride between patches (equal to patch_size for non-overlapping patches)
- `patch_level`: Pyramid level from which to extract patches (0 = highest resolution)

### Usage

```bash
python create_patches_fp.py --source path/to/manifest.json --save_dir RESULTS_DIRECTORY \
    --patch_size 1200 --step_size 1200 --preset preset_filename.csv --seg --patch --core 8
```

### Outputs

- **Patch Coordinates**:

  - Saves patch coordinates as .h5 files (one per slide)
  - Format allows on-the-fly loading during feature extraction
  - File naming is based on the slide name without file extension

- **Extracted Patches**:

  - Each slide has its own directory with extracted patch images
  - Patches are named according to their coordinates: `{x}_{y}_{size}.jpg`

- **Tissue Masks**:

  - Saves visualization of segmented tissue as JPG files
  - Useful for quality control and verification

- **Stitched Images (Optional)**:
  - Creates a downscaled visualization of all patches stitched together

### Multicore Processing

The code supports parallel processing of slides using multiple CPU cores:

- Use the `--core` parameter to specify the number of cores to use
- Each core can process a different slide simultaneously, significantly reducing total processing time
- Recommended to set this based on your system's available CPU cores and memory

### Output Directory Structure

```
RESULTS_DIRECTORY/
├── masks/                      # Tissue segmentation masks
│   ├── slide_1.jpg
│   └── ...
├── patches/                    # Contains patch coordinates and organized patch images
│   ├── slide_1.h5              # H5 file with patch coordinates
│   ├── slide_1/                # Directory containing extracted patches
│   │   ├── 0_0_1200.jpg        # Patches named as {x}_{y}_{size}.jpg
│   │   ├── 1200_0_1200.jpg
│   │   └── ...
│   ├── slide_2.h5
│   ├── slide_2/
│   │   └── ...
│   └── ...
├── stitches/                   # Visualization of stitched patches (optional)
│   ├── slide_1.jpg
│   └── ...
└── process_list_autogen.csv    # Processing parameters for each slide
```

## Resource Considerations

Processing large WSIs with high-resolution patches (1200×1200) requires significant computational resources. For production environments:

1. Use the `--core` parameter to enable parallel processing across multiple CPU cores
2. Ensure sufficient memory is available, especially when processing multiple slides simultaneously
3. Monitor disk space, as even storing just patch coordinates can consume substantial space with large datasets
4. When using multicore processing, be aware of potential I/O bottlenecks if all processes are reading/writing to the same disk
