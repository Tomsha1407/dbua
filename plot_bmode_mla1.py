import sys
import h5py
# import scipy.io
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from PIL import Image
from pathlib import Path
# from IPython.display import Image as IPythonImage
# from IPython.display import display
from dbua_us import channel_to_element


# from ..signal_biopsy_core.io.file_fetcher import fetch_ush5_file
# from sptb.data.data_utils import scan_conversion
# from sptb.consts import LOCAL_DIR, METADATA_DIR, IN_DIR, DID_EXAM_DIR, RAW_DIR, USH5_DIR
# from sptb.data.metadata_utils import (fetch_raws, fetch_exams, download_csv, get_bucket, 
#                                       decrypt_text, extract_image_data, fetch_ush5)

def load_bmode_ge(exam_id):
    exam_file_name = exam_id + ".png"
    exam_image_path = LOCAL_DIR / DID_EXAM_DIR / exam_file_name

    exam_image = Image.open(exam_image_path)

    # Resize image to make it smaller for display (e.g., width=800px, keep aspect ratio)
    base_width = 1000
    w_percent = (base_width / float(exam_image.size[0]))
    h_size = int((float(exam_image.size[1]) * float(w_percent)))
    exam_image_small = exam_image.resize((base_width, h_size), Image.LANCZOS)

    # Convert to numpy array for matplotlib display
    exam_image_array = np.array(exam_image_small)

    plt.figure(figsize=(8, h_size / 100))  # scale height to match image aspect
    plt.imshow(exam_image_array)
    plt.axis('off')
    plt.show()


from mla1 import bf_1frame

DATA_DIR = Path("./data")

def create_bmode_mla1(exp_name, ntx = None, nrx=None, nt = None):
    path_data = (f"{DATA_DIR}/{exp_name}.ush5")
    with h5py.File(path_data, 'r') as f:  
        fs = float(f["afe/[0]/sampling_rate_IQ"][0])
        fd = float(f["afe/[0]/demod_frequency"]['f_demod'][0,0])
        t0 = (np.array(f['channel_data']['[0]']['first_patient_sample'])[0]/fs) 
        tx_origin = np.array(f["tx_setup/[0]/origin"])  # Shape: (204,3) - effective TX sources
        elemnt_position = np.array(f["probe/element_positions"])  # Shape: (238,3) - RX elements
        rx_origin = np.array(f["rx_geometry/[0]/origin"]).T #(3, 816)  
        tx_directions = np.array(f["tx_setup/[0]/direction"]) #(203,3)   
        data = {}
        channel_data_list = []
        num_streams = 1 #6
        for i in range(num_streams):
            try:
                tmp_ch_data = f["hidden_data"]["channel_data"][f"[{i}]"]["channel_data_int16"]
            except KeyError:
                tmp_ch_data = f["hidden_data"]["channel_data"]["[0]"]["channel_data_int16"]
            ch_data = np.concatenate([
                np.expand_dims(np.array(tmp_ch_data[:, :, :]['i'], dtype=np.int16), axis=-1),
                np.expand_dims(np.array(tmp_ch_data[:, :, :]['r'], dtype=np.int16), axis=-1)
            ], axis=3)
            channel_data_list.append(ch_data)
        data["channel_data"] = np.stack(channel_data_list, axis=0)
        data["channel_element_mapping"] = np.stack([
            np.array(f["channel_data"][f"[{i}]"]["channel_element_map"]) for i in range(num_streams)
        ], axis=0)
    # change iqdata to elment_data. i.e arrange the cd correctly 

    element_data_from_CD = channel_to_element(channel_data=data["channel_data"], ch2el=data["channel_element_mapping"])
    iqdata_full = element_data_from_CD[0] #(204,238,1104) =(tx,rx,nt)
    # Remove rx padding symmetrically from center:
    num_elemnts = elemnt_position.shape[0]
    remove_rx = (iqdata_full.shape[1] - num_elemnts) // 2
    iqdata = iqdata_full[:,remove_rx:iqdata_full.shape[1]-remove_rx, :]
    
    if ntx != None:
        keep_tx = (iqdata.shape[0] - ntx) //2    
        iqdata = iqdata[keep_tx:keep_tx+ntx,:,:]
        tx_origin = tx_origin[keep_tx:keep_tx+ntx,:]
    if nrx != None:
        keep_rx = (iqdata.shape[1] - nrx) //2    
        iqdata = iqdata[:,keep_rx:keep_rx+nrx,:]
        elemnt_position = elemnt_position[keep_rx:keep_rx+nrx,:]

    if nt != None:
        iqdata = iqdata[:,:,:nt]
    
    data["el_data"] = iqdata
    data["tx_origins"] = tx_origin
    data["element_positions"] = elemnt_position
    data["tx_directions"] = tx_directions

    beamformed_image = bf_1frame(data)

    # log_abs_bf = np.log10(np.abs(beamformed_image[:, :980].T) + 1e-8)
    log_abs_bf = np.log10(np.abs(beamformed_image.T) + 1e-8)
    height, width = log_abs_bf.shape

    print(f"Original image shape: (y={height}, x={width})")
    print(f"Interpolated image shape: (y={log_abs_bf.shape[0]}, x={log_abs_bf.shape[1]})")

    ASSUMED_C = 1540
    wl0 = ASSUMED_C / fd  # wavelength (λ)
    BMODE_X_MIN = -(iqdata.shape[1]*wl0)/(6)
    BMODE_X_MAX = (iqdata.shape[1]*wl0)/(6)
    BMODE_Z_MIN = -(iqdata.shape[2]*wl0)/(6)
    BMODE_Z_MAX = (iqdata.shape[2]*wl0)/(6)

    # B-mode image dimensions
    xi = np.arange(BMODE_X_MIN, BMODE_X_MAX, wl0 / 3)
    zi = np.arange(BMODE_Z_MIN, BMODE_Z_MAX, wl0 / 3)
    nxi, nzi = xi.size, zi.size
    xi, zi = np.meshgrid(xi, zi, indexing="ij")
 
    xc = xi[:, 0] * 1e3
    y = zi[0, :] * 1e3
    figsize = (4, 6)
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_dpi(144)
    b_abs = np.abs(beamformed_image.T)
    bimg = b_abs / np.max(b_abs)
    bimg = bimg + 1e-10 * (bimg == 0)  # Avoid nans
    bimg = 20 * np.log10(bimg)
    dx = xc[1] - xc[0]
    dy = y[1] - y[0]
    ext = [xc[0] - dx / 2, xc[-1] + dx / 2, y[-1] + dy / 2, y[0] - dy / 2]
    # im = ax.imshow(bimg, vmin=-45, vmax=+5, extent=ext, cmap="bone",
    #                       interpolation="bicubic")
    im = ax.imshow(bimg, extent=ext, cmap="bone",
                          interpolation="bicubic")
    # plt.colorbar()
    plt.title("Beamformed Log-Abs Image")
    fig.savefig(f"mla/mla_woMinMax{exp_name}_ntx{ntx}_nrx{nrx}_nt{nt}.png", dpi=fig.get_dpi())

    # fig, ax = plt.subplots(figsize=figsize)
    # im = ax.imshow(log_abs_bf, cmap='gray', aspect='auto')
    # ax.set_title("Beamformed Log-Abs Image")
    # plt.axis('off')
    # plt.savefig(f"mla_{exp_name}_ntx{ntx}_nrx{nrx}_nt{nt}.png")
    # plt.show()

if __name__ == "__main__":
    exp_name = '0003490e_20250611'
    create_bmode_mla1(exp_name, ntx=100, nrx=100, nt=800)