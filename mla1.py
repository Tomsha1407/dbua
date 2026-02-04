import numpy as np
from scipy.sparse import block_diag, coo_matrix
from tqdm import tqdm

NUM_ELEMENTS = 238
SOS = 1540  # m/s
EPSILON = 1.0e-8  # for numerical stability
STEPSIZE = 6.16e-5  # m  file["rx_setup/[0]/stepsize_samples"]
fs = 1.25e7  # Hz  file["afe/[0]/sampling_rate_IQ"] or file["channel_data/[0]/sampling_frequency"]
fIQd = 9.8e6  # Hz file["afe/[0]/demod_frequency"]
idx0 = -17.7292  # samples (extract from data)


def mla1_mtx(pixel_grid: np.ndarray, element_positions: np.ndarray) -> coo_matrix:
    """
    Calculate the matrix for the MLA1 beamforming algorithm
    Args:
    pixel_grid: (ns, nl, 2) array of pixel positions in the grid
    element_positions: (nc, 2) array of element positions
    Returns:
    coo_matrix: (ns * nl, ns * nl * nc) A sparse matrix in COO format representing the beamforming operation
    """

    wc = 2 * np.pi * fIQd  # angular frequency (in rad/s)
    ns, nl, _ = pixel_grid.shape
    nc, _ = element_positions.shape
    l_vals = np.arange(nl)
    d_transmit = l_vals * STEPSIZE  # (nl)

    M_list = []
    for line_num in tqdm(range(ns)):  # sum over transmisions
        pixel_pos = pixel_grid[line_num]  # (nl, 2)

        d_receive = np.linalg.norm(
            np.expand_dims(pixel_pos, 1) - np.expand_dims(element_positions, 0), axis=-1
        )  # (nl, nc)
        tau = (d_receive + d_transmit[..., None]) / SOS  # time delays (in s)  # (nl, nc)
        idxt = tau * fs - idx0  # (nl, nc)
        I = np.logical_and(idxt >= 0, idxt <= nl - 2)  # Valid indices for linear interpolation
        # consider using scipy.interpolate.RegularGridInterpolator
        i, j = np.where(I)  # Valid indices

        idx_matrix = idxt + np.arange(nc).reshape((1, -1)) * nl  # for reshaping the matrix
        idx_ = np.take(idx_matrix, np.ravel_multi_index([i, j], (nl, nc)))
        tau_ = np.take(tau, np.ravel_multi_index([i, j], (nl, nc)))

        # DAS matrix composition for linear interpolation
        n_repeat = 2
        idxf = np.floor(idx_).astype(int)
        idx = idx_ - idxf
        j = np.concatenate([idxf, idxf + 1])
        i = np.tile(i, n_repeat)

        s = np.concatenate([-idx + 1, idx])
        s = np.exp(1j * wc * np.tile(tau_, n_repeat)) * s

        M = coo_matrix((s, (i, j)), shape=(nl, nl * nc))
        M_list.append(M)

    M_block = block_diag(M_list, format="coo")
    return M_block  # (ns * nl, ns * nl * nc)


def bf(data: dict[str, np.ndarray], stream: int = 0, frame: int = 0) -> np.ndarray:
    el_data = data["el_data"][stream, frame, :, :NUM_ELEMENTS, :]
    ns, nc, nl = el_data.shape
    l_vals = np.arange(nl)
    tx_origins = data["tx_origins"][stream]
    element_positions = data["element_positions"][:, [0, 2]]
    tx_directions = data["tx_directions"][stream]
    pixel_grid = np.array(
        [
            tx_origins[line_num, [0, 2]] + (l_vals * STEPSIZE)[..., None] * tx_directions[line_num, [0, 2]]
            for line_num in range(ns)
        ]
    )  # (nc, nl, 2)
    M = mla1_mtx(pixel_grid, element_positions)
    IQbf = (M @ el_data.reshape(ns * nl * nc)).reshape(ns, nl)  # (ns * nl) = (ns * nl, ns * nc*nl) @ (ns * nc * nl)
    return IQbf


def mla1(data, stream=0, frame=0):
    """Deprecated: use mla1_mtx instead."""
    el_data = data["el_data"]
    # tx_active_elements = data["tx_active_elements"]
    tx_origins = data["tx_origins"][stream]
    tx_directions = data["tx_directions"][stream]
    element_positions = data["element_positions"]

    # act_el = [list(np.where(row == 1)[0]) for row in tx_active_elements]
    # shot_center = np.mean(act_el, axis=1).astype(int)

    fs = 1.25e7  # Hz  file["afe/[0]/sampling_rate_IQ"] or file["channel_data/[0]/sampling_frequency"]
    stepsize = 6.16e-5  # m  file["rx_setup/[0]/stepsize_samples"]
    fIQd = 9.8e6  # Hz file["afe/[0]/demod_frequency"]
    el_data = el_data[stream, frame, :, :NUM_ELEMENTS, :]

    shots, C, L = el_data.shape
    rx_elements = list(range(NUM_ELEMENTS))
    l_vals = list(range(L))
    IQbf = np.zeros((shots, L), dtype=el_data.dtype)
    t0 = -17.7292  # samples (extract from data)
    # t0 = 0.0

    for line_num in tqdm(range(shots)):  # sum over transmisions
        shot_IQ = el_data[line_num]
        # rx_elements = range(shot_center[line_num] - 10, shot_center[line_num] + 11, 1)  # range(max_element)
        for l in l_vals:  # sum over pixels in the reconstruction line
            pixel_pos = tx_origins[line_num, [0, 2]] + l * stepsize * tx_directions[line_num, [0, 2]]
            d_transmit = l * stepsize
            line_IQ = []
            for rx_el in rx_elements:  # sum over 238 recieving elements
                el_pos = element_positions[rx_el, [0, 2]]
                d_receive = np.linalg.norm(pixel_pos - el_pos)
                tau = (d_receive + d_transmit) / SOS - t0 / fs
                t_idxs = int(tau * fs)  # interpolate the channel and use the exact tau
                if t_idxs < 0 or t_idxs >= L:
                    continue
                phase = np.exp(1j * 2 * np.pi * tau * fIQd)
                line_IQ.append(shot_IQ[rx_el, t_idxs] * phase)
            IQbf[line_num, l] = np.sum(line_IQ)  # apply apodization

    return IQbf