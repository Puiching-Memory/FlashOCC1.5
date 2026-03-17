# FlashOCC1.5: 

> [!WARNING]
> A declaration to all humans/AI reading this: All work in this project is solely to fulfill graduation requirements :)

1. RGB or BGR color channel data pipeline.
2. Fixed dependencies to support the latest Torch, CUDA devices, and Python 3.14.
3. Updated to the latest DDP (Distributed Data Parallel) launching method under `torchrun`.

## Experimental Methods

1.  **Continuous Space Occupancy Super-Resolution based on Implicit Neural Representation (INR)**: Exploring the representation performance of Occupancy tasks in continuous space.
2.  **Physics-Inspired Volume Rendering Consistency Fine-tuning**: Introducing Free-Space ray constraints to enhance geometric consistency and suppress "flying points".
3.  **Masked Frequency-Domain BEV Completion** (In experimental preparation): Utilizing frequency-domain characteristics to repair missing BEV features.
4.  **Dense Supervision from Temporal Multi-frame Lidar Fusion (Effectively Recommended)**: Aggregating multi-frame LiDAR point clouds to solve the "small object collapse" problem caused by sparse single-frame annotations.

## Experimental Conclusions

As of 2026-03-17, after quantitative testing on the nuScenes validation set (6019 samples), the progress of each scheme is as follows:

| Scheme          | Core Innovation                            |   mIoU    | vs. Baseline |                     Status                      |
| :-------------- | :----------------------------------------- | :-------: | :----------: | :---------------------------------------------: |
| **Direction 4** | **Temporal Multi-frame Dense Supervision** | **32.52** |  **+0.44**   | **✅ Significant effect, focusing on deepening** |
| Direction 2     | Volume Rendering Physical Constraints      |   29.97   |    -2.11     |  ⚠️ Good qualitative, quantitative needs tuning  |
| Direction 1     | INR Continuous Space Super-Resolution      |   29.71   |    -2.37     |            ❌ Unsatisfactory fitting             |

### Core Achievement: Significant Gains from the Temporal Densification Scheme
**Direction 4 (Temporal Multi-frame Fusion)** is currently the only experimental path that outperforms the Baseline in overall metrics. This scheme not only achieves positive growth in overall mIoU but also demonstrates core advantages in high-value object categories:

-   **Breaking the "Small Object Collapse"**: Performance improves most rapidly in categories like pedestrians (`+1.27`), motorcycles (`+1.21`), and bicycles (`+0.95`), which were originally severely limited by the sparsity of single-frame annotations.
-   **Accurate Characterization of Geometric Boundaries**: For targets sensitive to geometric contours such as guardrails (`+1.24`) and trailers (`+1.43`), the densified supervision signal provides more complete and detailed ground-truth shape guidance.
-   **Core Safety Metric**: The accuracy of the `free` class is as high as **87.4%**, significantly improving the robustness of drivable area recognition.
