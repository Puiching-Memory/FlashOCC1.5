# FlashOCC1.5

> [!WARNING]
> Statement for all humans/AIs reading this: All work in this project is only to satisfy graduation requirements :)  
> We have an engineering plan for FlashOCC called [FlashOCC2](https://github.com/Puiching-Memory/FlashOCC2), but we are currently facing some issues! We need your help!

---

## Improvements

1. RGB or BGR color channel data pipeline.
2. Fixed dependencies, compatible with the latest Torch, CUDA devices, and Python 3.14.
3. Updated to the latest DDP startup method under `torchrun`.

## Experimental Methods

1. **Continuous Space Occupancy Super-Resolution based on Implicit Neural Representation (INR)**: Exploring the representation performance of Occupancy tasks in continuous space.
2. **Physically-inspired Volume Rendering Consistency Fine-tuning**: Introducing Free-Space ray constraints to enhance geometric consistency and suppress "flying pixels."
3. **Masked Frequency-domain BEV Completion** (In preparation): Leveraging frequency-domain characteristics to repair missing BEV features.
4. **Dense Supervision via Temporal Multi-frame LiDAR Fusion (Highly Recommended)**: Aggregating multi-frame LiDAR point clouds to solve the "small object collapse" problem caused by sparse single-frame annotations.

## Experimental Conclusions

As of 2026-03-17, after quantitative testing on the nuScenes validation set (6,019 samples), the progress of each solution is as follows:

| Solution        | Core Innovation                            |   mIoU    | vs. Baseline |                    Status                     |
| :-------------- | :----------------------------------------- | :-------: | :----------: | :-------------------------------------------: |
| **Direction 4** | **Temporal Multi-frame Dense Supervision** | **32.52** |  **+0.44**   | **✅ Significant effect, focus for deepening** |
| Direction 2     | Volume Rendering Physical Constraints      |   29.97   |    -2.11     | ⚠️ Good qualitative, quantitative needs tuning |
| Direction 1     | INR Continuous Space Super-Resolution      |   29.71   |    -2.37     |           ❌ Unsatisfactory fitting            |

### Core Achievement: Significant Gains from Temporal Densification
**Direction 4 (Temporal Multi-frame Fusion)** is currently the only experimental path that outperforms the Baseline in overall metrics. This solution not only achieved positive growth in overall mIoU but also demonstrated core advantages in high-value object categories:

- **Breaking "Small Object Collapse"**: Performance improved most rapidly in categories like Pedestrians (`+1.27`), Motorcycles (`+1.21`), and Bicycles (`+0.95`), which were previously severely limited by the sparsity of single-frame annotations.
- **Precise Geometric Boundary Delineation**: For geometric-sensitive targets like Guardrails (`+1.24`) and Trailers (`+1.43`), the densified supervision signal provided more complete and detailed ground-truth shape guidance.
- **Key Safety Metric**: The accuracy of the `free` class reached **87.4%**, significantly improving the robustness of drivable area identification.
