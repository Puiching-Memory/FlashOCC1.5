import argparse
import os
import sys
import time
from contextlib import nullcontext

sys.path.insert(0, os.getcwd())

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint

from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
try:
    from mmdet.utils import compat_cfg, setup_multi_processes
except ImportError:
    from mmdet3d.utils import compat_cfg, setup_multi_processes
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.core.evaluation.occ_metrics import Metric_mIoU


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark and evaluate FlashOCC under different precisions"
    )
    parser.add_argument("config", help="config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--task",
        choices=["benchmark", "eval", "both"],
        default="both",
        help="which task to run",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="inference precision",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="single GPU id")
    parser.add_argument(
        "--benchmark-samples",
        type=int,
        default=200,
        help="number of samples for FPS benchmark",
    )
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=20,
        help="number of warmup samples for FPS benchmark",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="logging interval for benchmark",
    )
    parser.add_argument(
        "--eval-metric",
        nargs="+",
        default=["miou"],
        help="dataset evaluation metric list",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="limit evaluation samples, 0 means full set",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config options",
    )
    parser.add_argument(
        "--no-acceleration",
        action="store_true",
        help="disable pre-compute acceleration in view transformer",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=2,
        help="dataloader workers per GPU",
    )
    parser.add_argument(
        "--samples-per-gpu",
        type=int,
        default=1,
        help="dataloader batch size",
    )
    return parser.parse_args()


def import_plugin(cfg, config_path):
    if not getattr(cfg, "plugin", False):
        return
    import importlib

    if hasattr(cfg, "plugin_dir"):
        module_dir = os.path.dirname(cfg.plugin_dir)
    else:
        module_dir = os.path.dirname(config_path)
    module_path = module_dir.replace("/", ".")
    importlib.import_module(module_path)


def maybe_limit_dataset(cfg, max_eval_samples):
    if max_eval_samples <= 0:
        return
    ann_file = cfg.data.test.ann_file
    tmp_dir = os.path.join("work_dirs", "tmp_eval_ann")
    mmcv.mkdir_or_exist(tmp_dir)
    infos = mmcv.load(ann_file)
    if isinstance(infos, dict) and "infos" in infos:
        infos["infos"] = infos["infos"][:max_eval_samples]
        tmp_ann = os.path.join(
            tmp_dir,
            os.path.basename(ann_file).replace(".pkl", f".subset_{max_eval_samples}.pkl"),
        )
        mmcv.dump(infos, tmp_ann)
        cfg.data.test.ann_file = tmp_ann
    elif isinstance(infos, list):
        tmp_ann = os.path.join(
            tmp_dir,
            os.path.basename(ann_file).replace(".pkl", f".subset_{max_eval_samples}.pkl"),
        )
        mmcv.dump(infos[:max_eval_samples], tmp_ann)
        cfg.data.test.ann_file = tmp_ann
    else:
        raise TypeError(f"Unsupported ann file format: {type(infos)}")


def build_everything(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    import_plugin(cfg, args.config)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.gpu_ids = [args.gpu_id]

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.get("test_dataloader", {}).get("samples_per_gpu", 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    if not args.no_acceleration and "img_view_transformer" in cfg.model:
        cfg.model.img_view_transformer.accelerate = True

    maybe_limit_dataset(cfg, args.max_eval_samples)
    set_random_seed(args.seed, deterministic=False)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    elif hasattr(dataset, "PALETTE"):
        model.PALETTE = dataset.PALETTE

    model = model.cuda(args.gpu_id)
    model.eval()
    return cfg, dataset, data_loader, model


def get_autocast(precision):
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, list):
        return [_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def prepare_batch(data, device):
    batch = {}
    for key, value in data.items():
        if key == "img_inputs":
            batch[key] = [_to_device(value[0], device)]
        elif key in ("img_metas", "points"):
            batch[key] = value[0].data
            if key == "points":
                batch[key] = _to_device(batch[key], device)
        else:
            if isinstance(value, list) and len(value) > 0 and hasattr(value[0], "data"):
                batch[key] = _to_device(value[0].data, device)
            else:
                batch[key] = _to_device(value, device)
    return batch


def run_benchmark(args, data_loader, model):
    warmup = args.warmup_samples
    total = args.benchmark_samples
    pure_inf_time = 0.0

    for i, data in enumerate(data_loader):
        if i >= total:
            break
        batch = prepare_batch(data, args.gpu_id)

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            with get_autocast(args.precision):
                model(return_loss=False, rescale=True, **batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= warmup:
            pure_inf_time += elapsed
            done = i + 1 - warmup
            if (i + 1) % args.log_interval == 0:
                fps = done / pure_inf_time
                print(
                    f"[benchmark][{args.precision}] "
                    f"sample {i + 1}/{total}, fps={fps:.2f}, latency={1000.0 / fps:.2f} ms"
                )

    measured = max(0, min(total, i + 1) - warmup)
    fps = measured / pure_inf_time if pure_inf_time > 0 and measured > 0 else 0.0
    return {
        "precision": args.precision,
        "benchmark_samples": min(total, i + 1),
        "warmup_samples": warmup,
        "fps": fps,
        "latency_ms": (1000.0 / fps) if fps > 0 else float("inf"),
    }


def run_eval(args, cfg, dataset, data_loader, model):
    metric_name = args.eval_metric[0].lower()
    if metric_name == "miou":
        occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True,
        )
        print("metric = ", args.eval_metric[0])
        print("\nStarting Evaluation...")

        for i, data in enumerate(data_loader):
            batch = prepare_batch(data, args.gpu_id)
            with torch.no_grad():
                with get_autocast(args.precision):
                    result = model(return_loss=False, rescale=True, **batch)

            occ_pred = result[0] if isinstance(result, list) else result
            occ_pred = occ_pred["pred_occ"] if isinstance(occ_pred, dict) and "pred_occ" in occ_pred else occ_pred
            if torch.is_tensor(occ_pred):
                occ_pred = occ_pred.detach().cpu().numpy()

            info = dataset.data_infos[i]
            occ_gt = np.load(os.path.join(info["occ_path"], "labels.npz"))
            gt_semantics = occ_gt["semantics"]
            mask_lidar = occ_gt["mask_lidar"].astype(bool)
            mask_camera = occ_gt["mask_camera"].astype(bool)
            occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

            if (i + 1) % 200 == 0:
                print(f"[eval][{args.precision}] processed {i + 1}/{len(dataset)}", flush=True)

        return occ_eval_metrics.count_miou()

    outputs = []
    for i, data in enumerate(data_loader):
        batch = prepare_batch(data, args.gpu_id)
        with torch.no_grad():
            with get_autocast(args.precision):
                result = model(return_loss=False, rescale=True, **batch)
        if isinstance(result, list):
            outputs.extend(result)
        else:
            outputs.append(result)
        if (i + 1) % 50 == 0:
            print(f"[eval][{args.precision}] processed {i + 1}/{len(dataset)}")

    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=args.eval_metric))
    eval_results = dataset.evaluate(outputs, **eval_kwargs)
    return eval_results


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    cfg, dataset, data_loader, model = build_everything(args)

    if args.task in ("benchmark", "both"):
        bench = run_benchmark(args, data_loader, model)
        print(f"[summary][benchmark] {bench}")

    if args.task in ("eval", "both"):
        results = run_eval(args, cfg, dataset, data_loader, model)
        print(f"[summary][eval] {results}")


if __name__ == "__main__":
    main()
