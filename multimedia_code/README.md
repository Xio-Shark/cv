# Multimedia Experiments Codebase

This directory contains the **modularized and refactored code** for the multimedia experiments. It serves as the canonical implementation, replacing the legacy scattered scripts.

## 📂 Project Structure

The project follows a clean, modular architecture:

```
multimedia_code/
├── main.py                 # Unified entry point for all experiments
├── src/
│   ├── core/              # Core algorithms (Manual implementations)
│   │   ├── dct.py        # Discrete Cosine Transform (Matrix & Block)
│   │   ├── histogram.py  # Histogram Equalization & CLAHE
│   │   └── metrics.py    # PSNR, MSE calculations
│   │
│   ├── utils/             # Shared utilities
│   │   ├── io.py         # Image reading/saving
│   │   └── visualization.py # Plotting and graphing
│   │
│   └── experiments/       # Specific experiment logic
│       ├── exp1_1.py     # Histogram Enhancement
│       ├── exp1_2.py     # DCT Visualization
│       ├── exp2_1.py     # Block DC Compression
│       ├── exp2_2.py     # Global DC Compression
│       ├── exp2_3.py     # Top-K Compression (Fixed)
│       └── exp2_3_progressive.py # Top-K Compression (Progressive)
```

## 🚀 Usage Guide

Use the `main.py` script to run any experiment. It handles path setup and argument parsing.

### Basic Command

```bash
python main.py <experiment_id>
```

### Available Experiments

| ID                  | Description                                           |
| ------------------- | ----------------------------------------------------- |
| **1-1**             | Histogram Enhancement (Global HE & CLAHE)             |
| **1-2**             | DCT Transform & Visualization                         |
| **2-1**             | Block-based DCT Compression (DC only)                 |
| **2-2**             | Full-image DCT Compression (Global DC only)           |
| **2-3**             | Full-image DCT Compression (Top-K Coefficients)       |
| **2-3-progressive** | **Recommended**: Progressive DCT Compression Analysis |

### Options

- `--image <path>`: Specify input image (default: looks for `test_image.jpg`)
- `--output <dir>`: Specify output directory (default: `output`)

### Examples

```bash
# Run the progressive DCT experiment (Fast & Detailed)
python main.py 2-3-progressive

# Run histogram experiment on a specific image
python main.py 1-1 --image C:\Photos\test.jpg --output results_he
```

## 🧩 Design Principles

1.  **Modularity**: Algorithms are decoupled from experiment logic.
2.  **Consistency**: Uniform interfaces (`run(image_path, output_dir)`) and coding style.
3.  **Performance**: Core DCT algorithms use matrix multiplication for high speed.
4.  **Clarity**: All visualizations use English labels and clear layouts.
