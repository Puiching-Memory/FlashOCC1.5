import torch
from kernel import kernel_function
from problem import Model, build_case, get_inputs


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    if dtype == torch.float32:
        atol = 1e-4
        rtol = 1e-4
    else:
        atol = 3e-2
        rtol = 3e-2
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(f"max diff {diff} exceeds tolerance")


def _run_case(case):
    model = Model().cuda().eval()
    with torch.no_grad():
        ref = model(*case)
        out = kernel_function(*case)
    _assert_close(out, ref, case[0].dtype)


def test_kernel():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    base_case = [
        item.cuda() if isinstance(item, torch.Tensor) else item for item in get_inputs()
    ]
    _run_case(base_case)

    small_case = build_case(
        batch_size=1,
        cams=2,
        depth_bins=8,
        feat_h=4,
        feat_w=5,
        channels=32,
        bev_z=1,
        bev_y=16,
        bev_x=16,
        seed=17,
        dtype=torch.float32,
        device="cuda",
    )
    _run_case(list(small_case))

    print("PASS")
