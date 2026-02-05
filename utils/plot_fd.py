import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path("./data")

exp_name = '0003490e_20250611'
path_data = (f"{DATA_DIR}/{exp_name}.ush5")
with h5py.File(path_data, 'r') as f:
    print("Root keys:", list(f.keys()))
    # iqdata = jnp.array(f['channel_data']['[0]']['channel_data']) # should be tx,rx,nt
    t0 = 0 #-17.7292 #0 #??
    fs = float(f["afe/[0]/sampling_rate_IQ"][0])
    fd = float(f["afe/[0]/demod_frequency"]['f_demod'][0,0])
    test = np.array(f["afe/[0]/demod_frequency"]['f_demod'])
    print(test.max())
    print(test.min())
    # Load separate element position arrays for TX and RX
    elpos_tx = np.array(f["tx_setup/[0]/origin"]).T  # Shape: (3, 204) - effective TX sources
    elpos_rx = np.array(f["probe/element_positions"]).T  # Shape: (3, 238) - RX elements
    print(f"TX element positions shape: {elpos_tx.shape}")
    print(f"RX element positions shape: {elpos_rx.shape}")

    # plt.plot(test[0])
    # plt.savefig("fd.png")
    # plt.show()
    # print("end")

    t0_all = np.array(f['channel_data']['[0]']['first_patient_sample'])
    t0 = (np.array(f['channel_data']['[0]']['first_patient_sample'])[0]/fs) #0 #-17.7292 #0 

    print(t0)
