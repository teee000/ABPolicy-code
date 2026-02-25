import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import cv2
import os
import glob
import time
import shutil
import torchvision.transforms as T
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor


def _decode(buf):
    arr = np.frombuffer(buf, dtype=np.uint8)
    # BGR for Opencv;      RGB for torchvision
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)   
    # TODO
    # Opencv BGR   PyTorch   RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


NUM_THREADS = os.cpu_count()




class HDF5Dataset(Dataset):
    """
    Args
    ----
    roots : str | list[str] | list[pathlib.Path]
    """
    def __init__(
        self,
        roots,                                  
        qpos_delta_indices: list[int],
        action_delta_indices: list[int],
        cam_delta_indices: dict[str, list[int]]
    ):
        super().__init__()

        # -------- Normalize root input -------- #
        if isinstance(roots, (str, Path)):
            roots = [roots]                    
        self.roots = [Path(p).expanduser() for p in roots]

        # -------- Save delta indices -------- #
        self.delta_indices = {
            "qpos":   qpos_delta_indices,
            "action": action_delta_indices,
            **cam_delta_indices
        }
        self.camera_names = list(cam_delta_indices.keys())

        # -------- Scan all episode_* files -------- #
        self.episode_paths: list[Path] = []
        for r in self.roots:
            if not r.exists():
                raise FileNotFoundError(f"file not exists: {r}")
            self.episode_paths.extend(
                sorted(r.glob("episode_*.hdf5"))
            )

        if len(self.episode_paths) == 0:
            raise FileNotFoundError(
                f"No episode_*.hdf5 file found：\n" +
                "\n".join(str(p) for p in self.roots)
            )

        self.episode_paths = sorted(list({p.resolve() for p in self.episode_paths}))

        # -------- Read the length of each episode -------- #
        self.episode_lengths = []
        for path in self.episode_paths:
            with h5py.File(path, "r") as f:
                self.episode_lengths.append(f["action"].shape[0])

        # -------- Generate cumulative indices -------- #
        # Example: ep_lengths=[3,5,2] → cumulative=[0,3,8,10]
        self.cumulative_lengths = np.cumsum([0] + self.episode_lengths)
        self.total_length = int(self.cumulative_lengths[-1])


    def __len__(self):
        return self.total_length

    def _get_query_indices(self, query_idx: int, episode_len: int) -> tuple[dict, dict]:
        """
        Calculates the real indices and padding masks for all data types based on the given query_idx and episode length.
        This version uses a more robust method to create boolean tensors, completely eliminating DeprecationWarnings.
        """
        ep_start = 0
        ep_end = episode_len
        
        query_indices = {}
        padding_mask = {}

        for key, delta_idx_list in self.delta_indices.items():
            absolute_indices = [query_idx + delta for delta in delta_idx_list]
            
            clamped_indices = [max(ep_start, min(ep_end - 1, idx)) for idx in absolute_indices]
            query_indices[key] = clamped_indices
            
            is_pad = [(idx < ep_start) or (idx >= ep_end) for idx in absolute_indices]
            
            np_is_pad = np.array(is_pad, dtype=bool)
            padding_mask[f"{key}_is_pad"] = torch.from_numpy(np_is_pad)
            
        return query_indices, padding_mask
    

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_length:
            raise IndexError("Index out of range")

        episode_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        query_idx = idx - self.cumulative_lengths[episode_idx]
        
        episode_path = self.episode_paths[episode_idx]
        episode_len = self.episode_lengths[episode_idx]

        query_indices, padding_mask = self._get_query_indices(query_idx, episode_len)
        
        data_batch = {}

        with h5py.File(episode_path, 'r') as f:
            # --- process qpos & action ---
            for key in ['qpos', 'action']:
                target_indices = np.array(query_indices[key])
                h5_read_indices = np.unique(target_indices)
            
                h5_dataset_path = f'/observations/{key}' if key == 'qpos' else f'/{key}'
                data_from_h5 = f[h5_dataset_path][h5_read_indices]
                
                remap_indices = np.searchsorted(h5_read_indices, target_indices)
                data_batch[f'/observations/{key}'] = torch.from_numpy(data_from_h5[remap_indices]).float()

            # --- process image ---
            for cam_name in self.camera_names:
                target_indices = np.array(query_indices[cam_name])
                h5_read_indices = np.unique(target_indices)
                
                compressed_images_from_h5 = f[f'/observations/images/{cam_name}'][h5_read_indices]
                
                with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
                    decoded_images = list(pool.map(_decode, compressed_images_from_h5))
                
                decoded_images_np = np.stack(decoded_images)
                
                remap_indices = np.searchsorted(h5_read_indices, target_indices)
                final_images_np = np.ascontiguousarray(decoded_images_np[remap_indices])
                
                images_tensor = torch.from_numpy(final_images_np).permute(0, 3, 1, 2)
                data_batch[f'/observations/images/{cam_name}'] = images_tensor
                
        data_batch.update(padding_mask)
        return data_batch
    
    
    
    
    
    
