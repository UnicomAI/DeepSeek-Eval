# Quantitative Analysis of Performance Drop in DeepSeek Model Quantization


Enbo Zhao<sup>1,2</sup>, Yi Shen<sup>1,2</sup>, Shuming Shi<sup>1,2</sup>, Jieyun Huang<sup>1,2</sup>, Zhihao Chen<sup>1,2</sup>, Ning Wang<sup>1,2</sup>, Siqi Xiao<sup>1,2</sup>, Jian Zhang<sup>1,2</sup>, Kai Wang<sup>1,2</sup>, Shiguo Lian<sup>1,2</sup>

 
<sup>1</sup> Unicom Data Intelligence, China Unicom  
<sup>2</sup> Data Science & Artificial Intelligence Research Institute, China Unicom

## Paper Link

- **Arxiv**: [Quantitative Analysis of Performance Drop in DeepSeek Model Quantization](https://arxiv.org/pdf/2505.02390)


## Download Link

- **ModelScope**: [DeepSeek-DQ3_K_M](https://www.modelscope.cn/models/UnicomAI/DeepSeek-DQ3_K_M/)

## Abstract
Recently, there is a high demand for deploying DeepSeek-R1 and V3 locally, possibly because the official service often suffers from being busy and some organizations have data privacy concerns. While single-machine deployment offers infrastructure simplicity, the models’ 671B FP8 parameter configuration exceeds the practical memory limits of standard 8-GPU devices (A100/H100/910B). Quantization is a widely used technique that helps reduce model memory consumption. However, it is unclear what the performance of DeepSeek-R1 and V3 will be after being quantized. This technical report presents the first comprehensive evaluation of multibitwidth quantization across the complete DeepSeek model spectrum. Key findings reveal that 4-bit quantization maintains little performance degradation versus FP8 while enabling single-machine deployment on standard Nvidia GPU devices. We further propose DQ3_K_M, a dynamic 3-bit quantization method that significantly outperforms traditional Q3_K_M variantion various benchmarks, which is also comparable with 4-bit quantization (Q4_K_M) approach in most tasks. Moreover, DQ3_K_M supports single-machine deployment configurations for both NVIDIA H100/A100 and Huawei 910B. 

## Experimental Results

**Table 1:** Resource usage of DQ3_K_M versus llama.cpp and Unsloth quantizations for DeepSeek R1 (671B) at a 32K‑token context length.

| Metric                   | Q4_K_M (llama.cpp) | Q3_K_M (llama.cpp) | DQ3_K_M (ours) | Q2_K_L (llama.cpp) | UD‑Q2_K_XL (Unsloth) |
|--------------------------|--------------------|--------------------|----------------|--------------------|----------------------|
| Model Size               | 377G               | 298G               | 281G           | 228G               | 212G                 |
| Avg Quants               | 4.82               | 3.81               | 3.59           | 2.91               | 2.70                 |
| Total Memory Usage       | 568 GB             | 487 GB             | 469 GB         | 415 GB             | 398 GB               |
| Memory Usage per GPU     | 71 GB              | 61 GB              | 59 GB          | 52 GB              | 50 GB                |

<br>

**Table 2:** Quantization results of DeepSeek‑R1 on various benchmarks.

| Benchmark     | DeepSeek‑R1 FP8 (Reported) | FP8 (Official API) | Q4_K_M (llama.cpp) | Q3_K_M (llama.cpp) | UD‑Q2_K_XL (Unsloth) | DQ3_K_M (ours)  |
|---------------|-----------------------------|--------------------|--------------------|--------------------|----------------------|-----------------|
| AIME 2024     | 79.8                        | 77.53 (±2.97)      | 75.43 (±3.07)      | 72.50 (±6.11)      | 75.83 (±5.83)        | 75.41 (±4.69)   |
| MATH 500      | 97.3                        | 95.45 (±0.82)      | 95.55 (±0.44)      | 94.15 (±0.68)      | 95.25 (±0.44)        | 95.35 (±0.50)   |
| GPQA          | 71.5                        | 69.58 (±1.65)      | 69.95 (±1.85)      | 65.80 (±2.30)      | 68.93 (±1.55)        | 68.95 (±0.65)   |
| MBPP          | -                           | 92.60 (±0.80)      | 91.60 (±2.00)      | 90.43 (±0.88)      | 92.93 (±0.24)        | 92.80 (±0.70)   |
| MBPP+         | -                           | 78.35 (±1.06)      | 76.70 (±1.85)      | 76.75 (±0.88)      | 78.33 (±0.91)        | 78.60 (±1.01)   |
| LiveCodeBench | 65.9                        | 64.16 (±1.51)      | 62.41 (±2.27)      | 61.95 (±1.66)      | 61.40 (±1.59)        | 63.15 (±1.06)   |
| MMLU          | 90.8                        | 90.99              | 90.14              | 89.87              | 89.72                | 91.03           |
| CMMLU         | -                           | 90.37              | 90.42              | 89.85              | 89.61                | 90.17           |
| C‑Eval        | 91.8                        | 92.20              | 92.10              | 91.60              | 91.70                | 91.80           |
| **Average**   | -                           | 83.48              | 82.70              | 81.44              | 82.63                | 83.03           |
| Weighted avg. | -                           | 85.82              | 85.24              | 84.28              | 85.02                | 85.53           |
| Accuracy drop | -                           | -                  | 0.68%              | 1.80%              | 0.94%                | 0.34%           |

<br>

**Table 3:** Quantization results of DeepSeek-V3 on various benchmarks.

| Benchmark       | DeepSeek‑V3 FP8 (Reported)  | FP8 (Tencent API)    | Q4_K_M (llama.cpp)   | Q3_K_M (llama.cpp)   | Q2_K_L (llama.cpp)   | DQ3_K_M (ours)     |
|-----------------|-----------------------------|----------------------|----------------------|----------------------|----------------------|--------------------|
| AIME 2024       | 39.2                        | 38.34 (±2.52)        | 41.66 (±4.72)        | 38.73 (±4.70)        | 15.41 (±3.55)        | 39.16 (±4.97)      |
| MATH 500        | 90.2                        | 89.85 (±0.30)        | 90.55 (±0.44)        | 89.05 (±1.27)        | 77.30 (±0.66)        | 89.65 (±0.98)      |
| GPQA            | 59.1                        | 52.23 (±3.44)        | 51.95 (±2.64)        | 52.13 (±1.25)        | 43.65 (±1.32)        | 52.38 (±1.31)      |
| MBPP            | -                           | 87.75 (±0.61)        | 87.18 (±0.70)        | 88.55 (±0.90)        | 81.10 (±1.55)        | 89.38 (±0.35)      |
| MBPP+           | -                           | 73.35 (±1.21)        | 72.90 (±0.66)        | 73.08 (±1.31)        | 67.83 (±1.09)        | 74.78 (±0.56)      |
| LiveCodeBench   | 36.2                        | 36.21 (±0.47)        | 37.40 (±1.32)        | 36.21 (±2.03)        | 29.14 (±0.92)        | 36.76 (±0.67)      |
| MMLU            | 88.5                        | 88.06                | 88.09                | 87.31                | 84.25                | 87.87              |
| CMMLU           | -                           | 81.57                | 82.68                | 80.69                | 77.32                | 81.07              |
| C‑Eval          | 86.5                        | 83.10                | 82.90                | 82.60                | 77.60                | 83.40              |
| **Average**     | -                           | 70.05                | 70.59                | 69.82                | 61.51                | 70.47              |
| Weighted avg.   | -                           | 75.45                | 75.79                | 75.06                | 68.73                | 75.73              |
| Accuracy drop   | -                           | -                    | 0                    | 0.52%                | 8.91%                | 0                  |

<br>

**Table 4:** Quantization results of DeepSeek-R1-distill-Qwen-32B on various benchmarks

| Benchmark       | BF16 (Reported) | BF16 (Local Evaluation) | Q8_0 (llama.cpp)    | Q4_K_M (llama.cpp)   | Q3_K_M (llama.cpp)   |
|-----------------|-----------------|-------------------------|---------------------|----------------------|----------------------|
| AIME 2024       | 72.6            | 69.59 (±2.75)           | 71.68 (±4.71)       | 70.40 (±7.66)        | 71.24 (±6.66)        |
| MATH 500        | 94.3            | 93.65 (±0.41)           | 93.10 (±0.42)       | 93.90 (±0.53)        | 93.50 (±0.38)        |
| GPQA            | 62.1            | 61.85 (±2.18)           | 58.85 (±2.75)       | 62.00 (±4.54)        | 60.20 (±1.95)        |
| LiveCodeBench   | 57.2            | 57.08 (±1.01)           | 57.59 (±1.17)       | 56.85 (±2.87)        | 55.20 (±1.74)        |
| MBPP            | -               | 89.35 (±0.42)           | 89.35 (±0.73)       | 89.73 (±1.20)        | 88.93 (±0.64)        |
| MBPP+           | -               | 75.43 (±0.91)           | 75.45 (±1.18)       | 75.53 (±1.04)        | 75.38 (±1.30)        |
| MMLU            | -               | 82.15                   | 82.15               | 82.37                | 82.17                |
| CMMLU           | -               | 83.91                   | 83.97               | 83.57                | 83.34                |
| C‑Eval          | -               | 87.0                    | 86.7                | 86.8                 | 86.2                 |
| **Average**     | -               | 77.78                   | 77.65               | 77.91                | 77.35                |
| Weighted avg.   | -               | 79.94                   | 79.71               | 79.97                | 79.40                |
| Accuracy drop   | -               | -                       | 0.29%               | 0                    | 0.68%                |
