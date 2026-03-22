import argparse
import os
import sys

import torch


SOURCE_TO_TARGET_PREFIX = [
    ('model.0', 'stem.0'),
    ('model.1', 'stem.1'),
    ('model.2', 'stem.2'),
    ('model.3', 'stem.3'),
    ('model.4', 'stage0.branch_a'),
    ('model.5', 'stage0.branch_b'),
    ('model.6', 'stage0.convs.0'),
    ('model.7', 'stage0.convs.1'),
    ('model.8', 'stage0.convs.2'),
    ('model.9', 'stage0.convs.3'),
    ('model.11', 'stage0.transition'),
    ('model.13', 'trans1.conv_a'),
    ('model.14', 'trans1.conv_b1'),
    ('model.15', 'trans1.conv_b2'),
    ('model.17', 'stage1.branch_a'),
    ('model.18', 'stage1.branch_b'),
    ('model.19', 'stage1.convs.0'),
    ('model.20', 'stage1.convs.1'),
    ('model.21', 'stage1.convs.2'),
    ('model.22', 'stage1.convs.3'),
    ('model.24', 'stage1.transition'),
    ('model.26', 'trans2.conv_a'),
    ('model.27', 'trans2.conv_b1'),
    ('model.28', 'trans2.conv_b2'),
    ('model.30', 'stage2.branch_a'),
    ('model.31', 'stage2.branch_b'),
    ('model.32', 'stage2.convs.0'),
    ('model.33', 'stage2.convs.1'),
    ('model.34', 'stage2.convs.2'),
    ('model.35', 'stage2.convs.3'),
    ('model.37', 'stage2.transition'),
    ('model.39', 'trans3.conv_a'),
    ('model.40', 'trans3.conv_b1'),
    ('model.41', 'trans3.conv_b2'),
    ('model.43', 'stage3.branch_a'),
    ('model.44', 'stage3.branch_b'),
    ('model.45', 'stage3.convs.0'),
    ('model.46', 'stage3.convs.1'),
    ('model.47', 'stage3.convs.2'),
    ('model.48', 'stage3.convs.3'),
    ('model.50', 'stage3.transition'),
]

CONV_BN_SUFFIXES = [
    'conv.weight',
    'bn.weight',
    'bn.bias',
    'bn.running_mean',
    'bn.running_var',
    'bn.num_batches_tracked',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert official YOLOv7 training weights to FlashOCC ELAN backbone weights.')
    parser.add_argument('src', help='Path to the official yolov7_training.pt checkpoint')
    parser.add_argument('dst', help='Path to the converted ELAN backbone checkpoint')
    parser.add_argument(
        '--yolov7-repo',
        required=True,
        help='Path to a checkout of the official YOLOv7 repository used to deserialize the checkpoint')
    return parser.parse_args()


def load_source_state_dict(src_path, yolov7_repo):
    sys.path.insert(0, os.path.abspath(yolov7_repo))
    ckpt = torch.load(src_path, map_location='cpu', weights_only=False)
    model = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    if hasattr(model, 'float'):
        model = model.float()
    if hasattr(model, 'state_dict'):
        return model.state_dict()
    if isinstance(model, dict):
        return model
    raise TypeError(f'Unsupported checkpoint payload type: {type(model)}')


def convert_state_dict(source_state_dict):
    converted = {}
    for src_prefix, dst_prefix in SOURCE_TO_TARGET_PREFIX:
        for suffix in CONV_BN_SUFFIXES:
            src_key = f'{src_prefix}.{suffix}'
            dst_key = f'{dst_prefix}.{suffix}'
            if src_key not in source_state_dict:
                raise KeyError(f'Missing source key: {src_key}')
            converted[dst_key] = source_state_dict[src_key]
    return converted


def main():
    args = parse_args()
    source_state_dict = load_source_state_dict(args.src, args.yolov7_repo)
    converted = convert_state_dict(source_state_dict)
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save({'state_dict': converted, 'meta': {'source': args.src}}, args.dst)
    print(f'Saved {len(converted)} tensors to {args.dst}')


if __name__ == '__main__':
    main()