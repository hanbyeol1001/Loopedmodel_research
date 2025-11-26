import torch
import math


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Helper for RoPE: split last dim into even/odd and rotate.
    x: [..., 2 * n_rot]
    returns: [..., 2 * n_rot]
    """
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    # [a, b] -> [-b, a]
    x_rot_even = -x_odd
    x_rot_odd  = x_even
    x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)  # [..., n_rot, 2]
    x_rot = x_rot.flatten(-2)  # [..., 2 * n_rot]
    return x_rot


def apply_rope(x: torch.Tensor,
               cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE given precomputed cos/sin (already broadcastable to x).

    x:   [..., D]
    cos: [..., D]
    sin: [..., D]
    """
    x_rot = rotate_half(x)
    return x * cos + x_rot * sin


def build_2d_normalized_rope_cache(H: int,
                                   W: int,
                                   rope_dim: int,
                                   base: float = 10000.0,
                                   device=None):
    """
    Return:
        cos_x, sin_x, cos_y, sin_y each with shape [H, W, rope_dim]
    """
    assert rope_dim % 2 == 0, "rope_dim must be even (paired channels for rotate_half)"
    if device is None:
        device = torch.device("cpu")

    # normalized coords
    u = torch.linspace(0., 1., steps=H, device=device) if H > 1 else torch.tensor([0.0], device=device)
    v = torch.linspace(0., 1., steps=W, device=device) if W > 1 else torch.tensor([0.0], device=device)

    # use FULL rope_dim as rotating channels (not half!)
    n_rot = rope_dim  # <<< 중요: 절대 //2 하지 말 것

    inv_freq = 1.0 / (base ** (torch.arange(0, n_rot, 1, dtype=torch.float32, device=device) / n_rot))
    # angles
    theta_x = (u[:, None, None] * inv_freq[None, None, :]).expand(H, W, n_rot)  # [H, W, rope_dim]
    theta_y = (v[None, :, None] * inv_freq[None, None, :]).expand(H, W, n_rot)  # [H, W, rope_dim]

    cos_x = torch.cos(theta_x)
    sin_x = torch.sin(theta_x)
    cos_y = torch.cos(theta_y)
    sin_y = torch.sin(theta_y)

    return cos_x, sin_x, cos_y, sin_y



def apply_2d_normalized_rope_to_qk(q: torch.Tensor,
                                   k: torch.Tensor,
                                   H: int,
                                   W: int,
                                   base: float = 10000.0):
    """
    Apply 2D normalized RoPE (axial x/y) to q,k.

    Args:
        q, k: [B, n_heads, H*W, head_dim]
        H, W: maze dimensions
        base: RoPE frequency base

    Returns:
        q_rot, k_rot: same shape as q, k
    """
    B, n_heads, N, head_dim = q.shape
    assert N == H * W
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE (x/y split)."

    rope_dim = head_dim  # use full head dim for RoPE; you can choose smaller if you want
    half = rope_dim // 2  # channels for x, channels for y
    n_rot = half // 2     # #pairs for each axis

    device = q.device

    # reshape to grid
    q = q.view(B, n_heads, H, W, head_dim)
    k = k.view(B, n_heads, H, W, head_dim)

    # split into x-part and y-part
    q_x, q_y = q[..., :half], q[..., half:half*2]
    k_x, k_y = k[..., :half], k[..., half:half*2]

    # precompute cos/sin caches: [H, W, half]
    cos_x, sin_x, cos_y, sin_y = build_2d_normalized_rope_cache(
        H, W, rope_dim=half, base=base, device=device
    )
    
    # after building caches
    assert cos_x.shape[-1] == half and cos_y.shape[-1] == half, \
        f"RoPE cache last dim {cos_x.shape[-1]} must equal half={half}"

    
    # We need cos/sin to match q_x and q_y shapes: [B, n_heads, H, W, half]
    # Expand along batch and heads
    cos_x = cos_x[None, None, :, :, :]    # [1, 1, H, W, half]
    sin_x = sin_x[None, None, :, :, :]
    cos_y = cos_y[None, None, :, :, :]
    sin_y = sin_y[None, None, :, :, :]

    # To apply rotate_half, we need cos/sin shaped to [B, n_heads, H, W, half]
    cos_x = cos_x.expand(B, n_heads, H, W, half)
    sin_x = sin_x.expand(B, n_heads, H, W, half)
    cos_y = cos_y.expand(B, n_heads, H, W, half)
    sin_y = sin_y.expand(B, n_heads, H, W, half)

    # apply RoPE separately on x-channels and y-channels
    q_x_rot = apply_rope(q_x, cos_x, sin_x)
    k_x_rot = apply_rope(k_x, cos_x, sin_x)

    q_y_rot = apply_rope(q_y, cos_y, sin_y)
    k_y_rot = apply_rope(k_y, cos_y, sin_y)

    # concat back (you can keep any remaining channels if rope_dim < head_dim)
    q_rot = torch.cat([q_x_rot, q_y_rot], dim=-1)  # [B, n_heads, H, W, head_dim]
    k_rot = torch.cat([k_x_rot, k_y_rot], dim=-1)

    # reshape back to [B, n_heads, H*W, head_dim]
    q_rot = q_rot.view(B, n_heads, H * W, head_dim)
    k_rot = k_rot.view(B, n_heads, H * W, head_dim)

    return q_rot, k_rot