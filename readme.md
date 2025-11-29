# 快速使用指南

## 🎯 实验概览

本项目包含 5 个多媒体图像处理实验。

## 📦 环境安装

```bash
pip install numpy opencv-python matplotlib pillow
```

---

## 📂 项目结构

1.  **`multimedia_code/`**: 核心代码库。
2.  **`实验一/`, `实验二/`**: 实验运行入口。

**注意**: 所有实验结果都会统一生成在 **`multimedia_code/output/`** 对应的子文件夹下。

---

## 🚀 运行实验

直接进入对应的实验文件夹运行脚本即可。

### 1. 直方图增强 (实验一)

```bash
cd 实验一/直方图增强
python histogram_enhancement.py
```

**输出**: `multimedia_code/output/exp1_1_histogram/histogram_enhancement_result.png`

### 2. DCT 变换分析 (实验一)

```bash
cd 实验一/dct变换
python dct_transform.py
```

**输出**: `multimedia_code/output/exp1_2_dct_transform/dct_transform_result.png`

### 3. 8×8 分块 DCT (实验二-1)

```bash
cd 实验二/1
python dct_1.py
```

**输出**: `multimedia_code/output/exp2_1_block_dct/dct_result_visualization.png`

### 4. 整图 DCT (实验二-2)

```bash
cd 实验二/2
python dct_2.py
```

**输出**: `multimedia_code/output/exp2_2_global_dct/dct_whole_image_visualization.png`

### 5. 渐进式压缩 (实验二-3)

```bash
cd 实验二/3
python dct_3_progressive.py
```

**输出**: `multimedia_code/output/exp2_3_progressive/progressive_comparison.png`
