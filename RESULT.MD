## 实验结果 LaTeX 重写版

下面把原始 Markdown 表格改写为适合论文的 LaTeX 版本，并使用 `pgfplots` 绘制柱状图。

建议在导言区加入以下宏包：

```latex
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{pgfplots}
\usepackage{xcolor}
\pgfplotsset{compat=1.18}
```

实验设置：

```text
Batch Size: 16 | Num Workers: 0 | AMP Enabled: TRUE | Epochs: 5 | Cache Size Ratio: 0.3
```

### LaTeX 表格

```latex
\begin{table*}[t]
    \centering
    \caption{不同缓存策略在各主干网络上的耗时对比（单位：秒，越低越好）。仅保留 OPT (ours)、FIFO、LFU 和 None 四种策略。}
    \label{tab:cache_policy_results}
    \scriptsize
    \setlength{\tabcolsep}{5pt}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{lcccc}
        \toprule
        Model & OPT (ours) & FIFO & LFU & None \\
        \midrule
        convnextv2\_base              & 136.9226 & 179.1515 & 167.4298 & 195.6518 \\
        davit\_base                   &  82.9771 & 118.9020 & 118.8557 & 120.6464 \\
        mobilenetv3\_small\_100      &  47.3384 &  97.6476 &  95.3437 & 103.9736 \\
        mobilenetv5\_base             & 150.2256 & 168.7884 & 166.6462 & 189.0450 \\
        resnet50                      &  42.7169 &  65.3436 &  71.7038 &  84.1985 \\
        swin\_base\_patch4\_window7\_224 &  85.9536 & 126.8982 & 137.4744 & 125.5925 \\
        swinv2\_cr\_base\_224         & 104.4104 & 126.8298 & 156.9438 & 191.7603 \\
        vit\_base\_patch16\_224       &  67.5470 & 115.6438 &  98.5015 & 127.2664 \\
        vit\_base\_patch16\_dinov3    &  96.1758 &  94.6338 &  97.7933 & 160.4503 \\
        \bottomrule
    \end{tabular}%
    }
\end{table*}
```

### LaTeX 柱状图

```latex
\definecolor{fifoColor}{RGB}{78,121,167}
\definecolor{lfuColor}{RGB}{242,142,43}
\definecolor{optColor}{RGB}{118,183,178}
\definecolor{noneColor}{RGB}{176,122,161}

\begin{figure*}[t]
    \centering
    \begin{tikzpicture}
    \begin{axis}[
        ybar,
        width=\textwidth,
        height=8.2cm,
        bar width=5.5pt,
        xlabel={Backbone Model},
        ylabel={Time (s)},
        ymin=0,
        ymax=210,
        xtick={1,2,3,4,5,6,7,8,9},
        xticklabels={
            ConvNeXtV2-B,
            DaViT-B,
            MobileNetV3-S,
            MobileNetV5-B,
            ResNet50,
            Swin-B,
            SwinV2-B,
            ViT-B-16,
            ViT-B-DINOv3
        },
        xticklabel style={rotate=35, anchor=east, font=\scriptsize},
        yticklabel style={font=\scriptsize},
        legend style={
            at={(0.5,1.12)},
            anchor=south,
        legend columns=4,
        draw=none,
        font=\scriptsize
        },
        grid=major,
        grid style={dashed, gray!30},
        axis line style={gray!60},
        tick style={gray!60},
        enlarge x limits=0.05
    ]
    \addplot[fill=optColor, draw=optColor] coordinates {
        (1,136.9226) (2,82.9771) (3,47.3384) (4,150.2256) (5,42.7169)
        (6,85.9536) (7,104.4104) (8,67.5470) (9,96.1758)
    };
    \addplot[fill=fifoColor, draw=fifoColor] coordinates {
        (1,179.1515) (2,118.9020) (3,97.6476) (4,168.7884) (5,65.3436)
        (6,126.8982) (7,126.8298) (8,115.6438) (9,94.6338)
    };
    \addplot[fill=lfuColor, draw=lfuColor] coordinates {
        (1,167.4298) (2,118.8557) (3,95.3437) (4,166.6462) (5,71.7038)
        (6,137.4744) (7,156.9438) (8,98.5015) (9,97.7933)
    };
    \addplot[fill=noneColor, draw=noneColor] coordinates {
        (1,195.6518) (2,120.6464) (3,103.9736) (4,189.0450) (5,84.1985)
        (6,125.5925) (7,191.7603) (8,127.2664) (9,160.4503)
    };
    \legend{OPT (ours), FIFO, LFU, None}
    \end{axis}
    \end{tikzpicture}
    \caption{不同缓存策略在各主干网络上的推理耗时柱状图。仅保留 OPT (ours)、FIFO、LFU 和 None 四种策略。总体上，OPT (ours) 在大多数模型上具有最低耗时。}
    \label{fig:cache_policy_barplot}
\end{figure*}
```

### 独立编译版本

已额外提供一个可直接编译的完整 LaTeX 文件：

- `tex/cache_policy_results.tex`
