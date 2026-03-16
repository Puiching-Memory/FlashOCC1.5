## 环境配置
步骤 1. 安装用于 PyTorch 训练的环境
```bash
uv venv --python=3.14
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -U pip setuptools wheel ninja cython packaging

cd third_party/mmcv-1.5.3
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
uv pip install -v --no-build-isolation -e .

cd ..
cd nuscenes-devkit-1.2.0/setup
uv pip install --no-build-isolation -e . -v

cd ../..
cd mmdetection-2.25.1
uv pip install --no-build-isolation -e . -v

cd ..
cd mmsegmentation-0.25.0
uv pip install --no-build-isolation -e . -v

cd ..
cd mmdetection3d-1.0.0rc4
uv pip install --no-build-isolation -e . -v

cd ../..
uv pip install -r requirements.txt

cd ./projects
uv pip install --no-build-isolation -e . -v
```

步骤 3. 按照 [nuscenes_det.md](nuscenes_det.md) 中的说明准备 nuScenes 数据集，并执行以下命令为 FlashOCC 生成 pkl：
```shell
python tools/create_data_bevdet.py
```
完成后，目录结构如下：
```shell script
└── Path_to_FlashOcc/
    └── data
        └── nuscenes
            ├── v1.0-trainval (existing)
            ├── sweeps  (existing)
            ├── samples (existing)
            ├── bevdetv2-nuscenes_infos_train.pkl (new)
            └── bevdetv2-nuscenes_infos_val.pkl (new)
```

步骤 4. 对于 Occupancy Prediction 任务，从 [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction) 下载（仅）`gts`，并将目录组织为：
```shell script
└── Path_to_FlashOcc/
    └── data
        └── nuscenes
            ├── v1.0-trainval (existing)
            ├── sweeps  (existing)
            ├── samples (existing)
            ├── gts (new)
            ├── bevdetv2-nuscenes_infos_train.pkl (new)
            └── bevdetv2-nuscenes_infos_val.pkl (new)
```
（对于 panoptic occupancy），我们遵循 SparseOcc 的数据设置：

（1）从 [gdrive](https://drive.google.com/file/d/1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY/view?usp=drive_link) 下载 Occ3D-nuScenes occupancy GT，解压后保存到 `data/nuscenes/occ3d`。

（2）使用 `gen_instance_info.py` 生成 panoptic occupancy 标注。生成后的 panoptic 版本 Occ3D 将保存在 `data/nuscenes/occ3d_panoptic`。


步骤 5. 准备 CKPTS
（1）将 flashocc-r50-256x704.pth（[下载链接](https://drive.google.com/file/d/1k9BzXB2nRyvXhqf7GQx3XNSej6Oq6I-B/view)）下载到 Path_to_FlashOcc/FlashOcc/ckpts/，然后执行：
```shell script
bash tools/dist_test.sh projects/configs/flashocc/flashocc-r50.py  ckpts/flashocc-r50-256x704.pth 4 --eval map
```

步骤 6.（可选）安装 mmdeploy 用于 TensorRT 测试
```shell script
conda activate FlashOcc
pip install Cython==0.29.24

### 获取 tensorrt
wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/8.4.0/tars/TensorRT-8.4.0.6.Linux.x86_64-gnu.cuda-11.6.cudnn8.3.tar.gz
export TENSORRT_DIR=Path_to_TensorRT-8.4.0.6

### 获取 onnxruntime
ONNXRUNTIME_VERSION=1.8.1
pip install onnxruntime-gpu==${ONNXRUNTIME_VERSION}
cd Path_to_your_onnxruntime
wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
     && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
# export ONNXRUNTIME_DIR=/data01/shuchangyong/pkgs/onnxruntime-linux-x64-1.8.1
export ONNXRUNTIME_DIR=Path_to_your_onnxruntime/onnxruntime-linux-x64-1.8.1
cd Path_to_FlashOcc/FlashOcc/
git clone git@github.com:drilistbox/mmdeploy.git
cd Path_to_FlashOcc/FlashOcc/mmdeploy
git submodule update --init --recursive
mkdir -p build
cd Path_to_FlashOcc/FlashOcc/mmdeploy/build
cmake -DMMDEPLOY_TARGET_BACKENDS="ort;trt" ..
make -j 16
cd Path_to_FlashOcc/FlashOcc/mmdeploy
pip install -e .

### 构建 sdk
cd Path_to_pplcv/
git clone https://github.com/openppl-public/ppl.cv.git
cd Path_to_pplcv/ppl.cv
export PPLCV_VERSION=0.7.0
git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION}
./build.sh cuda

#pip install nvidia-tensorrt==8.4.0.6
pip install nvidia-tensorrt==8.4.1.5
pip install tensorrt
#pip install h5py
pip install spconv==2.3.6

export PATH=Path_to_TensorRT-8.4.0.6/bin:$PATH
export LD_LIBRARY_PATH=Path_to_TensorRT-8.4.0.6/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=Path_to_TensorRT-8.4.0.6/lib:$LIBRARY_PATH
```

## 最终整体目录结构
1. TensorRT
```shell script
└── Path_to_TensorRT-8.4.0.6
    └── TensorRT-8.4.0.6
```
2. FlashOcc
```shell script
└── Path_to_FlashOcc/
    └── data
        └── nuscenes
            ├── v1.0-trainval (existing)
            ├── sweeps  (existing)
            ├── samples (existing)
            ├── gts (new)
            ├── bevdetv2-nuscenes_infos_train.pkl (new)
            └── bevdetv2-nuscenes_infos_val.pkl (new)
    └── doc
        ├── install.md
        └── trt_test.md
    ├── figs
    ├── mmdeploy (new)
    ├── mmdetection3d (new)
    ├── projects
    ├── requirements
    ├── tools
    └── README.md
```
3. ppl.cv
```shell script
└── Path_to_pplcv
    └── ppl.cv
```
