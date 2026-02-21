import jax.numpy as jnp
from jax import jit, vmap, checkpoint
from jax.lax import map
from functools import partial
import numpy as np
from scipy.sparse import block_diag, coo_matrix
from mla1 import mla1_mtx
STEPSIZE = 6.16e-5  # m  file["rx_setup/[0]/stepsize_samples"]

def mla1_mtx_no_fixedSOS(pixel_grid: np.ndarray, element_positions: np.ndarray, fIQd: float, idx0: float, fs: float, sos: np.ndarray) -> jnp.ndarray:
    """
    Calculate the matrix for the MLA1 beamforming algorithm, using a SOS map instead of a fixed value.
    Args:
    pixel_grid: (ns, nl, 2) array of pixel positions in the grid
    element_positions: (nc, 2) array of element positions
    Returns:
    coo_matrix: (ns * nl, ns * nl * nc) A sparse matrix in COO format representing the beamforming operation
    """

    wc = 2 * jnp.pi * fIQd
    pixel_grid = jnp.asarray(pixel_grid)
    element_positions = jnp.asarray(element_positions)
    sos = jnp.asarray(sos)
    # sos = jnp.asarray(np.full_like(sos, 1540))

    ns, nl, _ = pixel_grid.shape
    nc, _ = element_positions.shape
    l_vals = jnp.arange(nl)
    d_transmit = l_vals * STEPSIZE  # (nl,)

    out_rows = ns * nl
    out_cols = ns * nl * nc
    M_block = jnp.zeros((out_rows, out_cols), dtype=jnp.complex32)

    # Build each per-line dense matrix and place into the block-diagonal positions
    for line_num in range(ns):
        pixel_pos = pixel_grid[line_num]  # (nl, 2)

        # distances from pixels to receive elements: (nl, nc)
        d_receive = jnp.linalg.norm(pixel_pos[:, None, :] - element_positions[None, :, :], axis=-1)
        tau = (d_receive + d_transmit[:, None]) / sos  # (nl, nc)
        idxt = tau * fs - idx0  # (nl, nc)

        # grid of row (pixel) and column (element) indices
        r_idx, c_idx = jnp.meshgrid(jnp.arange(nl), jnp.arange(nc), indexing="ij")

        r_flat = r_idx.ravel()
        c_flat = c_idx.ravel()
        idxt_flat = idxt.ravel()
        tau_flat = tau.ravel()

        valid = (idxt_flat >= 0) & (idxt_flat <= (nl - 2))
        if valid.sum() == 0:
            # no valid contributions for this line
            continue

        idxf = jnp.floor(idxt_flat).astype(int)
        frac = idxt_flat - idxf

        phase = jnp.exp(1j * wc * tau_flat)
        vals0 = (1 - frac) * phase
        vals1 = frac * phase

        cols0 = c_flat * nl + idxf
        cols1 = c_flat * nl + (idxf + 1)
        rows = r_flat

        # select only valid contributions
        rows_sel = rows[valid]
        cols0_sel = cols0[valid]
        cols1_sel = cols1[valid]
        vals0_sel = vals0[valid]
        vals1_sel = vals1[valid]

        # per-line dense matrix (nl, nl*nc)
        M_line = jnp.zeros((nl, nl * nc), dtype=jnp.complex128)
        M_line = M_line.at[rows_sel, cols0_sel].add(vals0_sel)
        M_line = M_line.at[rows_sel, cols1_sel].add(vals1_sel)

        row_off = line_num * nl
        col_off = line_num * nl * nc
        M_block = M_block.at[row_off : row_off + nl, col_off : col_off + nl * nc].set(M_line)

    return M_block

# def mla1_our(iqraw, tx_origins, element_positions, tx_directions, tA, tB, fs, fd, A=None, B=None, apoA=1, apoB=1, interp="cubic"):
def mla1_our(iqraw, tx_origins, element_positions, tx_directions,fd, t0,fs, c):
    """
    Operator-based MLA1 beamforming that computes the output directly without
    materializing the big matrix. This uses JAX primitives and `vmap` so it
    can be JITted and keeps memory low. Uses complex64 to reduce footprint.
    """
    iqraw = jnp.asarray(iqraw).astype(jnp.complex64)
    element_positions = jnp.asarray(element_positions)[:, [0, 2]]
    tx_origins = jnp.asarray(tx_origins)
    tx_directions = jnp.asarray(tx_directions)

    ns, nc, nl = iqraw.shape
    l_vals = jnp.arange(nl)
    d_transmit = l_vals * STEPSIZE  # (nl,)
    wc = 2 * jnp.pi * fd

    def process_line(iq_line, tx_origin, tx_dir):
        # iq_line: (nc, nl)
        # build pixel positions for this line: (nl, 2)
        # Avoid list-style multidimensional indexing on JAX arrays
        txo = jnp.array([tx_origin[0], tx_origin[2]])
        txd = jnp.array([tx_dir[0], tx_dir[2]])
        pixel_pos = txo + (l_vals * STEPSIZE)[:, None] * txd

        # distances (nl, nc)
        d_receive = jnp.linalg.norm(pixel_pos[:, None, :] - element_positions[None, :, :], axis=-1)
        # tau and fractional sample indices (nl, nc)
        tau = (d_receive + d_transmit[:, None]) / c.T
        idxt = tau * fs - t0

        # prepare for gather: transpose iq_line to (nl, nc) so axis 0 is sample index
        iq_line_samples = iq_line.T  # (nl, nc)

        idxf = jnp.floor(idxt).astype(int)
        frac = idxt - idxf

        # clip indices for safe gathering then mask invalid entries
        idxf_clipped = jnp.clip(idxf, 0, nl - 1)
        idxf1_clipped = jnp.clip(idxf + 1, 0, nl - 1)

        s0 = jnp.take_along_axis(iq_line_samples, idxf_clipped, axis=0)
        s1 = jnp.take_along_axis(iq_line_samples, idxf1_clipped, axis=0)

        phase = jnp.exp(1j * wc * tau).astype(jnp.complex64)
        contrib = ((1.0 - frac) * s0 + frac * s1) * phase

        valid = (idxt >= 0) & (idxt <= (nl - 2))
        contrib = contrib * valid.astype(jnp.complex64)

        # sum over receive elements -> per-pixel value (nl,)
        out_line = jnp.sum(contrib, axis=1)
        return out_line

    IQbf = vmap(process_line, in_axes=(0, 0, 0))(iqraw, tx_origins, tx_directions)
    return IQbf

@partial(jit, static_argnums=(3, 4))
def das(iqraw, tA, tB, fs, fd, A=None, B=None, apoA=1, apoB=1, interp="cubic"):
    """
    Delay-and-sum IQ data according to a given time delay profile.
    @param iqraw   [na, nb, nsamps]  Raw IQ data (baseband)
    @param tA      [na, *pixdims]    Time delays to apply to dimension 0 of iq
    @param tB      [nb, *pixdims]    Time delays to apply to dimension 1 of iq
    @param fs      scalar            Sampling frequency to convert from time to samples
    @param fd      scalar            Demodulation frequency (0 for RF modulated data)
    @param A       [*na_out, na]     Linear combination of dimension 0 of iqraw
    @param B       [*nb_out, nb]     Linear combination of dimension 1 of iqraw
    @param apoA    [na, *pixdims]    Broadcastable apodization on dimension 0 of iq
    @param apoB    [nb, *pixdims]    Broadcastable apodization on dimension 1 of iq
    @param interp  string            Interpolation method to use
    @return iqfoc  [*na_out, *nb_out, *pixel_dims]   Beamformed IQ data

    The tensors A and B specify how to combine the "elements" in dimensions 0 and 1 of
    iqraw via a tensor contraction. If A or B are None, they default to a vector of ones,
    i.e. a simple sum over all elements. If A or B are identity matrices, the result will
    be the delayed-but-not-summed output. A and B can be arbitrary tensors of arbitrary
    size, as long as the inner most dimension matches na or nb, respectively. Another
    alternative use case is for subaperture beamforming.

    Note that via acoustic reciprocity, it does not matter whether a or b correspond to
    the transmit or receive "elements".
    """
    # The default linear combination is to sum all elements.
    if A is None:
        A = jnp.ones((iqraw.shape[0],))
    if B is None:
        B = jnp.ones((iqraw.shape[1],))

    # Choose the interpolating function
    fints = {
        "nearest": interp_nearest,
        "linear": interp_linear,
        "cubic": interp_cubic,
        "lanczos3": lambda x, t: interp_lanczos(x, t, nlobe=3),
        "lanczos5": lambda x, t: interp_lanczos(x, t, nlobe=5),
    }
    fint = fints[interp]

    # Baseband interpolator
    def bbint(iq, t):
        iqfoc = fint(iq, fs * t)
        return iqfoc * jnp.exp(2j * jnp.pi * fd * t)

    # # Delay-and-sum beamforming (vmap inner, vmap outer)
    # # This method uses vmap to push both the inner and outer loops into XLA, which uses
    # # uses more memory, but can take advantage of XLA's parallelization.  However, it is
    # # slower when memory bandwidth is a bottleneck.
    # def das_b(iq_i, tA_i):
    #     return jnp.tensordot(B, vmap(bbint)(iq_i, tA_i + tB) * apoB, (-1, 0))
    # return jnp.tensordot(A, vmap(das_b)(iqraw, tA) * apoA, (-1, 0))

    # Delay-and-sum beamforming (vmap inner, map outer)
    # This method does not vmap the outer loop and thus cannot take advantage of XLA's
    # parallelization. However, it uses less memory and is faster when memory bandwidth
    # is a bottleneck.
    @checkpoint
    def das_b(x):
        iq_i, tA_i = x
        return jnp.tensordot(B, vmap(bbint)(iq_i, tA_i + tB) * apoB, (-1, 0))

    return jnp.tensordot(A, map(das_b, (iqraw, tA)) * apoA, (-1, 0))


def safe_access(x: jnp.ndarray, s):
    """Safe access to array x at indices s.
    @param x: Array to access
    @param s: Indices to access at
    @return: Array of values at indices s
    """
    s = s.astype("int32")
    valid = (s >= 0) & (s < x.size)
    return jnp.where(valid, jnp.where(valid, x[s], 0), 0)


def interp_nearest(x: jnp.ndarray, si: jnp.ndarray):
    """1D nearest neighbor interpolation with jax.
    @param x: 1D array of values to interpolate
    @param si: Indices to interpolate at
    @return: Interpolated signal
    """
    return x[jnp.clip(jnp.round(si), 0, x.shape[0] - 1).astype("int32")]


def interp_linear(x: jnp.ndarray, si: jnp.ndarray):
    """1D linear interpolation with jax.
    @param x: 1D array of values to interpolate
    @param si: Indices to interpolate at
    @return: Interpolated signal
    """
    f, s = jnp.modf(si)  # Extract fractional, integer parts
    x0 = safe_access(x, s + 0)
    x1 = safe_access(x, s + 1)
    return (1 - f) * x0 + f * x1


def interp_cubic(x: jnp.ndarray, si: jnp.ndarray):
    """1D cubic Hermite interpolation with jax.
    @param x: 1D array of values to interpolate
    @param si: Indices to interpolate at
    @return: Interpolated signal
    """
    f, s = jnp.modf(si)  # Extract fractional, integer parts
    # Values
    x0 = safe_access(x, s - 1)
    x1 = safe_access(x, s + 0)
    x2 = safe_access(x, s + 1)
    x3 = safe_access(x, s + 2)
    # Coefficients
    a0 = 0 + f * (-1 + f * (+2 * f - 1))
    a1 = 2 + f * (+0 + f * (-5 * f + 3))
    a2 = 0 + f * (+1 + f * (+4 * f - 3))
    a3 = 0 + f * (+0 + f * (-1 * f + 1))
    return (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3) / 2


def _lanczos_helper(x, nlobe=3):
    """Lanczos kernel"""
    a = (nlobe + 1) / 2
    return jnp.where(jnp.abs(x) < a, jnp.sinc(x) * jnp.sinc(x / a), 0)


def interp_lanczos(x: jnp.ndarray, si: jnp.ndarray, nlobe=3):
    """Lanczos interpolation with jax.
    @param x: 1D array of values to interpolate
    @param si: Indices to interpolate at
    @param nlobe: Number of lobes of the sinc function (e.g., 3 or 5)
    @return: Interpolated signal
    """
    f, s = jnp.modf(si)  # Extract fractional, integer parts
    x0 = safe_access(x, s - 1)
    x1 = safe_access(x, s + 0)
    x2 = safe_access(x, s + 1)
    x3 = safe_access(x, s + 2)
    a0 = _lanczos_helper(f + 1, nlobe)
    a1 = _lanczos_helper(f + 0, nlobe)
    a2 = _lanczos_helper(f - 1, nlobe)
    a3 = _lanczos_helper(f - 2, nlobe)
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3
