
#### 训练模型
```shell
# 单卡
python tools/train.py $config
# 多卡
./tools/dist_train.sh $config num_gpu
```

颜色通道支持通过 `data_config.channel_order` 配置（可选 `BGR` / `RGB`），
当前官方配置默认使用 `BGR` 以兼容历史训练权重。

#### 测试模型
```shell
# 单卡
python tools/test.py $config $checkpoint --eval mAP
# 多卡
./tools/dist_test.sh $config $checkpoint num_gpu --eval mAP
# ray-iou 指标
./tools/dist_test.sh $config $checkpoint num_gpu --eval ray-iou
```

#### Panoptic-FlashOcc 的 FPS 测试
```shell
# 单帧
python tools/analysis_tools/benchmark.py  config ckpt 
python tools/analysis_tools/benchmark.py  config ckpt --w_pano

# 多帧
python tools/analysis_tools/benchmark_sequential.py  config ckpt 
python tools/analysis_tools/benchmark_sequential.py  config ckpt --w_pano
```
