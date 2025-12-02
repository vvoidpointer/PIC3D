import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
matplotlib.use('Agg')

def load_all_particles_from_all_files():
    """
    读取所有时刻的粒子数据，并返回一个列表
    
    Returns:
    - all_particle_data: 列表，每个元素是一个二维numpy数组，代表一个时刻的粒子数据
      每个数组的形状为 (n_particles, 17)
      列的顺序为：id, x, y, z, x_old, y_old, px, py, pz, sx, sy, sz, gamma, charge, mass, weight, rank
    - field_names: 列名列表
    """
    files = sorted(glob.glob('output/particles_step_*.h5'))
    if not files:
        print("错误：在 'output/' 目录下未找到 'particles_step_*.h5' 文件")
        return None, None
        
    all_particle_data = []
    field_names = ['id', 'x', 'y', 'z', 'x_old', 'y_old', 'px', 'py', 'pz', 
                   'sx', 'sy', 'sz', 'gamma', 'charge', 'mass', 'weight', 'rank']
                   
    for filename in files:
        print(f"正在读取文件: {filename}")
        try:
            with h5py.File(filename, 'r') as f:
                particles = f['particles'][:]
                n_particles = len(particles)
                
                # 检查文件是否为空
                if n_particles == 0:
                    print(f"警告: 文件 {filename} 不包含粒子数据，跳过")
                    continue
                    
                particle_data_at_step = np.zeros((n_particles, len(field_names)))
                
                for i, field in enumerate(field_names):
                    particle_data_at_step[:, i] = particles[field]
                
                all_particle_data.append(particle_data_at_step)
                print(f"成功读取 {n_particles} 个粒子的数据")

        except Exception as e:
            print(f"读取文件 {filename} 时发生错误: {e}")
            
    if not all_particle_data:
        print("错误：未能成功读取任何粒子数据")
        return None, None
        
    return all_particle_data, field_names

if __name__ == "__main__":
    # 读取所有时刻的粒子数据
    all_particle_data, field_names = load_all_particles_from_all_files()
    
    if all_particle_data:
        num_timesteps = len(all_particle_data)
        print(f"\n成功读取 {num_timesteps} 个时刻的粒子数据")
        
        # 示例：访问零时刻的数据
        time_step_index = 20
        if time_step_index < num_timesteps:
            particle_data_t0 = all_particle_data[time_step_index]
            print(f"\n--- 时刻 {time_step_index} 的数据统计 ---")
            print(f"粒子总数: {particle_data_t0.shape[0]}")
            
            # 显示前几个粒子的数据
            print(f"\n前5个粒子的数据:")
            print("Index\t" + "\t".join([f"{name:8s}" for name in field_names]))
            for i in range(min(5, particle_data_t0.shape[0])):
                values = "\t".join([f"{particle_data_t0[i, j]:8.3f}" for j in range(len(field_names))])
                print(f"{i}\t{values}")
        
        print(f"\n数据已存储在列表 all_particle_data 中")
        print(f"您可以使用以下方式访问数据:")
        print(f"- all_particle_data[0]          # 零时刻的所有粒子数据")
        print(f"- all_particle_data[0][:, 1]    # 零时刻所有粒子的x坐标")
        print(f"- all_particle_data[1][:, 6]    # 第一个时刻所有粒子的px动量")
        print(f"字段索引对照: {dict(zip(field_names, range(len(field_names))))}")