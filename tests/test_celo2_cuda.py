"""Numerical-alignment tests for the CUDA CELO2 / ELO-CELO2 optimizers.

We validate the CUDA implementation against the pure-PyTorch naive one. These
tests establish agreement with that reference for the exercised inputs; they do
not independently establish higher-rank equivalence to JAX/meta-training.

Both optimizers are fed the *same* meta-model weights (``network=net``), the same
initial parameters, and the same gradient sequence; we then compare the parameter
trajectories step by step. Rank-4 regressions also compare raw kernel updates,
before orthogonalization and final normalization can conceal feature errors.
Their deterministic fixed networks complement the separately required parity
check with the pinned learned weights; they never fetch a checkpoint.

These tests require a CUDA device and the compiled ``celo2_cuda_kernel``
extension; they skip otherwise.
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CELO2 CUDA optimizer requires a GPU"
)

# Skip cleanly if the extension hasn't been built.
celo2_cuda_kernel = pytest.importorskip("celo2_cuda_kernel")

from pylo.models.CELO2_MLP import CELO2MLP
from pylo.optim.CELO2_naive import CELO2_naive
from pylo.optim.ELO_CELO2_naive import ELO_CELO2_naive
from pylo.optim.CELO2_cuda import CELO2_CUDA
from pylo.optim.ELO_CELO2_cuda import ELO_CELO2_CUDA

DEVICE = torch.device("cuda")

# Matrix params in both orientations (R<C and R>C, to exercise both dc/dr index
# paths) plus a 1D param to exercise the shared-accumulator AdamW branch.
SHAPES = [(64, 32), (32, 64), (32,)]


def _make_init(seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return [torch.randn(s, generator=g, device=DEVICE) for s in SHAPES]


def _make_grads(n_steps, seed, scale=3.0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return [
        [torch.randn(s, generator=g, device=DEVICE) * scale for s in SHAPES]
        for _ in range(n_steps)
    ]


def _run(opt_class, init_vals, grads_seq, net, **cfg):
    params = [torch.nn.Parameter(v.clone()) for v in init_vals]
    opt = opt_class(params, network=net, **cfg)
    traj = []
    for grads in grads_seq:
        for p, gg in zip(params, grads):
            p.grad = gg.clone()
        opt.step()
        traj.append([p.detach().clone() for p in params])
    return traj


def _max_traj_diff(traj_a, traj_b):
    worst = 0.0
    for step_a, step_b in zip(traj_a, traj_b):
        for a, b in zip(step_a, step_b):
            worst = max(worst, torch.max(torch.abs(a - b)).item())
    return worst


@pytest.mark.parametrize("orthogonalize", [True, False])
@pytest.mark.parametrize("exp_mult", [0.0, 0.01])
def test_celo2_cuda_matches_naive(orthogonalize, exp_mult):
    torch.manual_seed(0)
    np.random.seed(0)
    net = CELO2MLP().to(DEVICE)  # shared random-but-fixed meta-model

    init_vals = _make_init(seed=1)
    grads_seq = _make_grads(n_steps=50, seed=2)
    cfg = dict(
        lr=1e-2,
        weight_decay=0.1,
        clip_grad=True,
        clip_norm=1.0,
        orthogonalize=orthogonalize,
        exp_mult=exp_mult,
    )

    naive_traj = _run(CELO2_naive, init_vals, grads_seq, net, **cfg)
    cuda_traj = _run(CELO2_CUDA, init_vals, grads_seq, net, **cfg)

    assert _max_traj_diff(naive_traj, cuda_traj) < 1e-5


def test_elo_celo2_cuda_matches_naive():
    """ELO-CELO2 (different defaults, shared-accumulator AdamW) stays aligned."""
    torch.manual_seed(0)
    np.random.seed(0)
    net = CELO2MLP().to(DEVICE)

    init_vals = _make_init(seed=3)
    grads_seq = _make_grads(n_steps=50, seed=4)
    # The CUDA ELO class defaults to adam_lr_mult=20, while the naive class
    # defaults to 1. Match the settings explicitly, including the vector branch.
    cfg = dict(
        lr=1e-3,
        weight_decay=0.1,
        adam_lr_mult=1.0,
        adam_weight_decay=0.1,
        clip_grad=True,
        clip_norm=1.0,
    )

    naive_traj = _run(ELO_CELO2_naive, init_vals, grads_seq, net, **cfg)
    cuda_traj = _run(ELO_CELO2_CUDA, init_vals, grads_seq, net, **cfg)

    assert _max_traj_diff(naive_traj, cuda_traj) < 1e-5


# Factoring selects prefix, mixed, or spatial axes across these cases. Every
# case retains nonunit residual coordinates in the factored buffers.
RANK4_SHAPES = [
    pytest.param((7, 5, 2, 3), id="prefix-row-largest-small-groups"),
    pytest.param((5, 7, 3, 2), id="prefix-col-largest-small-groups"),
    pytest.param((5, 5, 2, 3), id="tied-largest-prefix-axes"),
    pytest.param((7, 2, 3, 5), id="mixed-prefix-spatial-factors"),
    pytest.param((2, 3, 7, 5), id="spatial-factors-warp-tail"),
    pytest.param((3, 2, 17, 19), id="spatial-groups-block-tail"),
]
OPTIMIZER_PAIRS = [
    pytest.param(CELO2_naive, CELO2_CUDA, id="celo2"),
    pytest.param(ELO_CELO2_naive, ELO_CELO2_CUDA, id="elo-celo2"),
]


@pytest.fixture
def rank4_fp32(monkeypatch):
    """Use deterministic PyTorch operations and full-precision FP32 matmuls."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    cudnn_deterministic = torch.backends.cudnn.deterministic
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _rank4_inputs(shape):
    """Unequal prefix/spatial scales expose residual-index and grouping errors."""
    g = torch.Generator(device=DEVICE).manual_seed(1729)
    o, i, h, w = [
        torch.linspace(0.25, 2.0, size, device=DEVICE).reshape(
            [size if axis == dim else 1 for axis in range(4)]
        )
        for dim, size in enumerate(shape)
    ]
    init = (
        torch.randn(shape, generator=g, device=DEVICE)
        * (0.1 + o * i) * (h + w) * 0.1
    )
    grads = [
        [
            (torch.randn(shape, generator=g, device=DEVICE) + 0.2 * (-1) ** step)
            * (0.2 + o * i)
            * (0.3 + h * w)
        ]
        for step in range(8)
    ]
    return [init], grads


def _rank4_config(orthogonalize, exp_mult):
    # Keep both classes on the same explicit settings, without changing their
    # public defaults. The reference remains the actual naive optimizer path.
    return dict(
        lr=1e-3,
        weight_decay=0.1,
        adam_lr_mult=1.0,
        adam_weight_decay=0.1,
        use_adamw_for_1d=True,
        clip_grad=True,
        clip_norm=1.0,
        orthogonalize=orthogonalize,
        initial_momentum_decays=(0.9, 0.99, 0.999),
        initial_rms_decays=(0.95,),
        initial_adafactor_decays=(0.9, 0.99, 0.999),
        exp_mult=exp_mult,
        rmsmult=1.0,
        param_scale_mult=False,
        ns_coeffs=(3.4445, -4.7750, 2.0315),
        ns_iters=5,
        ns_eps=1e-8,
        grad_clip_val=1000.0,
    )


def _rank4_alignment(naive_class, cuda_class, init, grads, net, cfg, monkeypatch):
    reference_raw = []
    actual_raw = []

    def capture_reference(module, inputs, output):
        # Use the actual naive feature construction and network, not a second
        # implementation of either the reduction or factored-index formulas.
        reference_raw.append(
            (output[..., 0] * torch.exp(output[..., 1] * cfg["exp_mult"]))
            .detach().clone()
        )

    handle = net.register_forward_hook(capture_reference)
    try:
        reference = _run(naive_class, init, grads, net, **cfg)
    finally:
        handle.remove()

    kernel = celo2_cuda_kernel.celo2_kernel

    def capture_kernel(*args):
        result = kernel(*args)
        # The public extension writes argument 15 before Python postprocessing.
        actual_raw.append(args[15].detach().clone())
        return result

    monkeypatch.setattr(celo2_cuda_kernel, "celo2_kernel", capture_kernel)
    actual = _run(cuda_class, init, grads, net, **cfg)
    assert len(reference_raw) == len(actual_raw) == len(grads) == 8
    previous_reference, previous_actual = init, init
    for step, (expected, observed) in enumerate(zip(reference, actual), start=1):
        context = f"{cuda_class.__name__}, shape={tuple(init[0].shape)}, step={step}"
        torch.testing.assert_close(
            actual_raw[step - 1], reference_raw[step - 1],
            rtol=1e-4, atol=1e-6, msg=f"raw MLP update: {context}",
        )
        for p_ref, p_cuda, prev_ref, prev_cuda in zip(
            expected, observed, previous_reference, previous_actual
        ):
            torch.testing.assert_close(
                p_cuda, p_ref, rtol=1e-4, atol=1e-6, msg=f"parameter: {context}",
            )
            torch.testing.assert_close(
                p_cuda - prev_cuda, p_ref - prev_ref,
                rtol=1e-3, atol=1e-6, msg=f"update delta: {context}",
            )
        previous_reference, previous_actual = expected, observed


@pytest.mark.parametrize("naive_class,cuda_class", OPTIMIZER_PAIRS)
@pytest.mark.parametrize("shape", RANK4_SHAPES)
@pytest.mark.parametrize("orthogonalize,exp_mult", [(False, 0.0), (True, 0.01)])
def test_celo2_cuda_rank4_matches_naive(
    naive_class, cuda_class, shape, orthogonalize, exp_mult, rank4_fp32, monkeypatch
):
    """Exercise all feature channels and all three decay planes for eight updates."""
    torch.manual_seed(314159)
    net = CELO2MLP().to(DEVICE)
    init, grads = _rank4_inputs(shape)
    _rank4_alignment(
        naive_class, cuda_class, init, grads, net,
        _rank4_config(orthogonalize, exp_mult), monkeypatch,
    )


@pytest.mark.parametrize("naive_class,cuda_class", OPTIMIZER_PAIRS)
@pytest.mark.parametrize("shape", RANK4_SHAPES)
def test_celo2_cuda_rank4_gradient_channel_normalization(
    naive_class, cuda_class, shape, rank4_fp32, monkeypatch
):
    """Isolate spatial normalization from factor-buffer geometry.

    This fixed network outputs 32 + normalized_g. Since every spatial group has
    at most 323 elements, normalized_g has magnitude at most sqrt(323). The
    positive offset therefore keeps both ReLUs linear, so only the gradient
    feature determines the varying direction.
    In particular, scaling a whole prefix differently cannot be normalized away
    accidentally by the final update normalization before the raw comparison.
    """
    net = CELO2MLP().to(DEVICE)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.w0[0][0, 0] = 1.0
        net.b0[0] = 32.0
        net.dense_w[0][0, 0] = 1.0
        net.dense_w[1][0, 0] = 1.0
    init, grads = _rank4_inputs(shape)
    _rank4_alignment(
        naive_class, cuda_class, init, grads, net,
        _rank4_config(False, 0.0), monkeypatch,
    )


def test_celo2_cuda_single_matrix_alignment():
    """Tight check on a single matrix param (pure CELO2 path, no AdamW)."""
    torch.manual_seed(0)
    np.random.seed(0)
    net = CELO2MLP().to(DEVICE)

    g = torch.Generator(device=DEVICE).manual_seed(7)
    init = [torch.randn(48, 24, generator=g, device=DEVICE)]
    grads = [[torch.randn(48, 24, generator=g, device=DEVICE) * 2.0] for _ in range(50)]
    cfg = dict(lr=1e-2, orthogonalize=True)

    naive_traj = _run(CELO2_naive, init, grads, net, **cfg)
    cuda_traj = _run(CELO2_CUDA, init, grads, net, **cfg)

    assert _max_traj_diff(naive_traj, cuda_traj) < 1e-5
