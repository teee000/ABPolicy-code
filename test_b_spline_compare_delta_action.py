import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline, BSpline
from omegaconf import OmegaConf


from utils.hdf5_dataloader import HDF5Dataset


class BSplineFitter:
    def __init__(self, T: int, k: int = 3, n_ctrl: int = 6):
        self.T = T
        self.k = k
        self.n_ctrl = n_ctrl
        self.x = np.arange(T, dtype=float)

        t_internal = np.linspace(self.x[0], self.x[-1], n_ctrl - k + 1)
        self.t = np.concatenate(([self.x[0]] * k, t_internal, [self.x[-1]] * k)).astype(float)

    def fit(self, y: np.ndarray):
        spline = make_lsq_spline(self.x, y, self.t, self.k, axis=0)
        return spline.c.astype(np.float32)

    def rebuild(self, ctrl_y: np.ndarray):
        spline = BSpline(self.t, ctrl_y, self.k, extrapolate=False)
        return spline(self.x).astype(np.float32)


def to_cumulative_delta(seq):
    return seq - seq[0]

def from_cumulative_delta(cum_seq, p0):
    return cum_seq + p0

def to_incremental_delta(seq):
    inc_seq = np.zeros_like(seq)
    inc_seq[1:] = seq[1:] - seq[:-1]
    return inc_seq

def from_incremental_delta(inc_seq, p0):
    return np.cumsum(inc_seq, axis=0) + p0



if __name__ == '__main__':

    DATA_PATH = '/media/yf/CODE/TeleOperation/my_teleoperation-lerobot-jianhua2-cage/record_data/hole_insert_soft'
    
    
    T = 40 
    action_indices = [i for i in range(T)]
    
    dataset = HDF5Dataset(
        roots=DATA_PATH,
        qpos_delta_indices=[0], 
        action_delta_indices=action_indices,
        cam_delta_indices={} 
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=1,
        batch_size=1,
        shuffle=True,      
        pin_memory=False,
        drop_last=True,
    )
    
    
    k = 3              
    n_ctrl = 8         
    fitter = BSplineFitter(T=T, k=k, n_ctrl=n_ctrl)

    max_test_samples = 100  
    num_tested = 0
    
    sum_err_abs = 0.0
    sum_err_cum = 0.0
    sum_err_inc = 0.0

    for idx, batch in enumerate(dataloader):
        abs_seq = batch['/observations/action'].cpu().squeeze().numpy()
        if abs_seq.shape[0] != T:
            continue  

        p0 = abs_seq[0]  

        
        cum_seq = to_cumulative_delta(abs_seq)
        inc_seq = to_incremental_delta(abs_seq)

        
        fit_abs = fitter.rebuild(fitter.fit(abs_seq))
        fit_cum = fitter.rebuild(fitter.fit(cum_seq))
        fit_inc = fitter.rebuild(fitter.fit(inc_seq))

       
        rec_from_abs = fit_abs
        rec_from_cum = from_cumulative_delta(fit_cum, p0)
        rec_from_inc = from_incremental_delta(fit_inc, p0)

        
        sum_err_abs += np.abs(rec_from_abs - abs_seq).mean()
        sum_err_cum += np.abs(rec_from_cum - abs_seq).mean()
        sum_err_inc += np.abs(rec_from_inc - abs_seq).mean()

        num_tested += 1
        if num_tested >= max_test_samples:
            break

    avg_err_abs = sum_err_abs / num_tested
    avg_err_cum = sum_err_cum / num_tested
    avg_err_inc = sum_err_inc / num_tested

    print("\n" + "="*50)
    print(f" Test NUm: {num_tested}")
    print(f"Bspline k={k}, n_ctrl={n_ctrl}, action len T={T}")
    print(f"=== Error (MAE) ===")
    print(f"1. Absolute Actions MAE:         {avg_err_abs:.6f}")
    print(f"2. Cumulative Delta MAE:         {avg_err_cum:.6f}")
    print(f"3. Incremental Delta MAE: {avg_err_inc:.6f}")
    print("="*50 + "\n")


    dim_to_plot = 0
    t_axis = np.arange(T)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"Real Data: B-Spline Fitting Comparison (Sample {num_tested}, Dim {dim_to_plot})", fontsize=14)

    # Plot 1: Absolute
    axs[0].plot(t_axis, abs_seq[:, dim_to_plot], 'k.-', label='Raw (GT)')
    axs[0].plot(t_axis, rec_from_abs[:, dim_to_plot], 'r-', lw=2, label='Fitted')
    axs[0].set_title(f"1. Absolute Actions")
    axs[0].legend(); axs[0].grid(True, alpha=0.3)

    # Plot 2: Cumulative Delta
    axs[1].plot(t_axis, abs_seq[:, dim_to_plot], 'k.-', label='Raw (GT)')
    axs[1].plot(t_axis, rec_from_cum[:, dim_to_plot], 'g-', lw=2, label='Fitted & Reconstructed')
    axs[1].set_title(f"2. Cumulative Delta")
    axs[1].legend(); axs[1].grid(True, alpha=0.3)

    # Plot 3: Incremental Delta 
    axs[2].plot(t_axis, abs_seq[:, dim_to_plot], 'k.-', label='Raw (GT)')
    axs[2].plot(t_axis, rec_from_inc[:, dim_to_plot], 'b-', lw=2, label='Fitted & Reconstructed')
    axs[2].set_title(f"3. Incremental Delta - Notice the trajectory drift")
    axs[2].set_xlabel("Time step (action horizon)")
    axs[2].legend(); axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()