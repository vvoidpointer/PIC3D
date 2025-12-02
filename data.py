

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


P_flag = 1  # 是否显示 PIC3D



if P_flag == 1:
    # 定义要显示的时间步
    time_steps = [240,260,280,300]
    
    # 创建子图 (1行4列)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes = axes.flatten()
    
    for i, step in enumerate(time_steps):
        filename = f'output/output_step_{step:04d}.h5'
        
        try:
            with h5py.File(filename, 'r') as h5file:
                data = h5file['ey'][()]  # data已经是NumPy数组    
            
            if data.ndim == 3:
                # 三维数据 [z][y][x]，取中间Z平面的切片
                nz, ny, nx = data.shape
                matrix = data[:, ny//2, :]
                a = matrix # shape is (ny, nx), correct for imshow with origin='lower'
                title_suffix = f' (Z={nz//2})'
            else:
                # 二维数据 [x][y]
                matrix = np.array(data)
                a = matrix.T
                title_suffix = ''
            
            im = axes[i].imshow(a, extent=[0, 20, 0, 20], cmap='bwr', origin='lower',vmin=-0.5,vmax=0.5)
            axes[i].set_title(f'Step {step}{title_suffix}')
            axes[i].set_xlabel('X')

            if i == 0:  # 只为第一个子图添加Y标签
                axes[i].set_ylabel('Y')
            
            # 为每个子图添加颜色条，调整尺寸
            cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
            cbar.ax.tick_params(labelsize=8)
            
        except FileNotFoundError:
            axes[i].text(0.5, 0.5, f'File not found:\n{filename}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'Step {step} - Missing')
    
    # 隐藏空余的子图
    if len(time_steps) < len(axes):
        axes[len(time_steps)].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('rho_electron_overview.png', dpi=300)