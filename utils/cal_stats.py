import os
import glob
import json
from typing import Sequence, Union, List

import h5py
import numpy as np
from tqdm import tqdm


def calculate_hdf5_statistics(
        data_dirs: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
        output_json_path: str
) -> dict | None:
    """
    读取一个或多个目录中所有 HDF5 文件里的 qpos 与 action 数据，
    计算每个维度的最小值和最大值，并将结果保存为 JSON。

    Args
    ----
    data_dirs : str | list[str]
        单个目录路径或目录路径序列。
    output_json_path : str
        输出 JSON 文件完整路径。

    Returns
    -------
    dict | None
        统计结果；若失败返回 None。
    """

    # ---------- 1. 标准化输入，把它变成列表 ----------
    if isinstance(data_dirs, (str, os.PathLike)):
        data_dirs = [data_dirs]

    # ---------- 2. 过滤非法目录 ----------
    valid_dirs: List[str] = []
    for d in data_dirs:
        if os.path.isdir(d):
            valid_dirs.append(os.fspath(d))
        else:
            print(f"警告: 路径“{d}”不是有效目录，已忽略。")

    if not valid_dirs:
        print("错误: 未提供任何有效目录。")
        return None

    # ---------- 3. 收集所有 .hdf5 文件 ----------
    hdf5_files: List[str] = []
    for d in valid_dirs:
        hdf5_files.extend(glob.glob(os.path.join(d, "*.hdf5")))

    if not hdf5_files:
        print("未在指定目录中找到任何 .hdf5 文件。")
        return None

    # ---------- 4. 逐文件读取并收集数据 ----------
    all_qpos_data, all_action_data = [], []
    for filepath in tqdm(hdf5_files, desc="处理 HDF5 文件"):
        filename = os.path.basename(filepath)
        try:
            with h5py.File(filepath, "r") as f:
                # 读取 qpos
                if "observations" in f and "qpos" in f["observations"]:
                    all_qpos_data.append(f["observations"]["qpos"][:])
                else:
                    tqdm.write(f"警告: 文件“{filename}”缺少 /observations/qpos，已跳过此数据。")

                # 读取 action
                if "action" in f:
                    all_action_data.append(f["action"][:])
                else:
                    tqdm.write(f"警告: 文件“{filename}”缺少 /action，已跳过此数据。")

        except Exception as e:
            tqdm.write(f"错误: 处理文件“{filename}”时发生异常: {e}")

    # ---------- 5. 统计 ----------
    statistics_results: dict = {"processed_files_count": len(hdf5_files)}

    # qpos
    if all_qpos_data:
        combined_qpos = np.concatenate(all_qpos_data, axis=0)
        statistics_results["qpos"] = {
            "min": np.min(combined_qpos, axis=0).tolist(),
            "max": np.max(combined_qpos, axis=0).tolist(),
        }
        print("\n--- qpos 每维最小/最大值 ---")
        print("min:", [f"{v:.6f}" for v in statistics_results["qpos"]["min"]])
        print("max:", [f"{v:.6f}" for v in statistics_results["qpos"]["max"]])
    else:
        print("\n没有任何 qpos 数据可供统计。")

    # action
    if all_action_data:
        combined_action = np.concatenate(all_action_data, axis=0)
        statistics_results["action"] = {
            "min": np.min(combined_action, axis=0).tolist(),
            "max": np.max(combined_action, axis=0).tolist(),
        }
        print("\n--- action 每维最小/最大值 ---")
        print("min:", [f"{v:.6f}" for v in statistics_results["action"]["min"]])
        print("max:", [f"{v:.6f}" for v in statistics_results["action"]["max"]])
    else:
        print("\n没有任何 action 数据可供统计。")

    # ---------- 6. 保存 ----------
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(statistics_results, f, indent=4, ensure_ascii=False)
        print(f"\n统计结果已保存至 {output_json_path}")
    except Exception as e:
        print(f"错误: 写入 JSON 失败: {e}")
        return None

    return statistics_results


if __name__ == "__main__":
    
    HDF5_DATA_DIR = '/media/yf/CODE/TeleOperation/my_teleoperation-lerobot-jianhua2-cage/record_data' 
    
    HDF5_DATA_DIR = [
        '/media/yf/CODE/TeleOperation/my_teleoperation-lerobot-jianhua2-cage/record_data',
        '/media/yf/CODE/TeleOperation/my_teleoperation-lerobot-jianhua2-cage/record_data_two_obj'
        ]
    
    OUTPUT_JSON_FILE = './dataset_stats.json' # 输出 JSON 文件的名称
    stats = calculate_hdf5_statistics(HDF5_DATA_DIR, OUTPUT_JSON_FILE)
    print(stats)














