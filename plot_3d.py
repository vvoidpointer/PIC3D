import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

# 使用 Agg 后端，不显示窗口直接保存图片
matplotlib.use('Agg')

def plot_3d_scatter(step, field_name='rho_electron', threshold_std=2.0, vmin=None, vmax=None):
    filename = f'output/output_step_{step:04d}.h5'
    output_img = f'3D_view_{field_name}_{step:04d}.png'
    
    print(f"Reading {field_name} from {filename}...")
    try:
        with h5py.File(filename, 'r') as h5file:
            if field_name not in h5file:
                print(f"Error: Field '{field_name}' not found. Keys: {list(h5file.keys())}")
                return
            data = h5file[field_name][()]
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return

    if data.ndim != 3:
        print("Error: Data is not 3D.")
        return

    # data shape is [z, y, x]
    nz, ny, nx = data.shape
    print(f"Data shape: {data.shape}")

    # 计算阈值
    # 对于电磁场 (Ex, Ey, Ez...)，通常关心绝对值较大的区域
    # 对于密度 (rho)，通常关心数值较大的区域
    if field_name in ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'jx', 'jy', 'jz', 'rho_electron']:
        data_for_threshold = np.abs(data)
        threshold = np.mean(data_for_threshold) + threshold_std * np.std(data_for_threshold)
        print(f"Threshold (Abs): {threshold:.4e} (Mean + {threshold_std}*Std)")
        z_idx, y_idx, x_idx = np.where(data_for_threshold > threshold)
    else:
        threshold = np.mean(data) + threshold_std * np.std(data)
        print(f"Threshold: {threshold:.4e} (Mean + {threshold_std}*Std)")
        z_idx, y_idx, x_idx = np.where(data > threshold)

    values = data[z_idx, y_idx, x_idx]

    print(f"Number of points to plot: {len(values)}")
    
    if len(values) == 0:
        print("No points above threshold found. Try lowering the threshold_std.")
        return

    # 为了防止点太多导致绘图极慢或内存溢出，可以进行降采样
    max_points = 50000
    if len(values) > max_points:
        print(f"Downsampling from {len(values)} to {max_points} points...")
        indices = np.random.choice(len(values), max_points, replace=False)
        z_idx = z_idx[indices]
        y_idx = y_idx[indices]
        x_idx = x_idx[indices]
        values = values[indices]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    # 使用 bwr (blue-white-red) colormap 适合展示正负值分布的场
    cmap = 'bwr' if field_name in ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'jx', 'jy', 'jz'] else 'jet'
    
    # 如果是场数据且未指定范围，自动设置对称范围以便于观察
    if field_name in ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'jx', 'jy', 'jz'] and vmin is None and vmax is None:
        max_val = np.max(np.abs(values))
        vmin = -max_val
        vmax = max_val
    
    # 生成全局唯一的颜色映射，确保每个整数值(1, 2...40000, 40001)都有独特的颜色
    from matplotlib.colors import ListedColormap, hsv_to_rgb
    
    # 1. 确定数据范围 (假设为正整数ID)
    # 使用整个数据的最大值来生成颜色表，确保颜色索引正确
    max_val_int = int(np.nanmax(data))
    
    # 2. 生成颜色表
    n_colors = max_val_int + 1
    indices = np.arange(n_colors)
    
    # Golden Angle Chaos: Hue = (index * 0.618...) % 1.0
    golden_ratio_conjugate = 0.618033988749895
    h = (indices * golden_ratio_conjugate) % 1.0
    s = 0.6 + (indices * 0.23) % 0.4  # Saturation 0.6 - 1.0
    v = 0.85 + (indices * 0.13) % 0.15  # Value 0.85 - 1.0 (保持明亮)
    
    colors = hsv_to_rgb(np.column_stack((h, s, v)))
    # 可选：将 0 设为白色背景
    colors[0] = [1, 1, 1] 
    
    custom_cmap = ListedColormap(colors)
    
    # 更新 vmin/vmax 以匹配 ID 范围
    vmin = 0
    vmax = max_val_int

    # 增加 alpha (不透明度) 和 s (点大小) 让颜色更深更明显
    p = ax.scatter(x_idx, y_idx, z_idx, c=values, cmap=custom_cmap, s=2, alpha=0.8, vmin=vmin, vmax=vmax)

    ax.set_xlabel('X Grid')
    ax.set_ylabel('Y Grid')
    ax.set_zlabel('Z Grid')
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    
    # 设置合适的视角 (elevation, azimuth)
    ax.view_init(elev=30, azim=-60)
    
    ax.set_title(f'3D {field_name} (Step {step})\nThreshold > Mean + {threshold_std}*Std')
    fig.colorbar(p, ax=ax, label=field_name, shrink=0.6)

    print(f"Saving figure to {output_img}...")
    plt.savefig(output_img, dpi=150)
    plt.close()
    print("Done.")


if __name__ == "__main__":
    # 你可以在这里修改要查看的时间步
    steps_to_plot = [20,40,60,80,100]
    
    for s in steps_to_plot:
        # 你可以在这里手动指定 vmin 和 vmax 来固定颜色范围
        # 例如: plot_3d_scatter(s, field_name='ey', threshold_std=3.0, vmin=-0.5, vmax=0.5)
        # rho_electron 是负值，且分布可能比较稀疏或均匀，降低阈值标准差以捕获更多点
        plot_3d_scatter(s, field_name='jx', threshold_std=1.0)
