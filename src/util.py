import json
import os
import random
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from typing import Tuple, Union
import imageio
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit import mesh
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from IPython.display import Image, display
from pyeit.eit.fem import EITForward, Forward
from pyeit.eit.interp2d import pdegrad, sim2pts
from PIL import Image
from scipy.integrate import cumulative_trapezoid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from tqdm import tqdm
from .classes import (
    BallAnomaly,
    Boundary,
)


def define_mesh_obj(n_el,use_customize_shape):
    
    n_el = 16 
    use_customize_shape = False
    
    if use_customize_shape:
        mesh_obj = mesh.create(n_el, h0=0.05, fd=thorax)
    else:
        mesh_obj = mesh.create(n_el, h0=0.05)
                  
    return mesh_obj

def plot_mesh(mesh_obj, figsize: tuple = (6, 4), title: str = "mesh") -> None:
  
    plt.style.use("default")
    pts = mesh_obj.node
    tri = mesh_obj.element
    x, y = pts[:, 0], pts[:, 1]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.tripcolor(
        x,
        y,
        tri,
        np.real(mesh_obj.perm_array),
        edgecolors="k",
        shading="flat",
        alpha=0.5,
        cmap=plt.cm.viridis,
    )

    ax.plot(x[mesh_obj.el_pos], y[mesh_obj.el_pos], "ro")
    for i, e in enumerate(mesh_obj.el_pos):
        ax.text(x[e], y[e], str(i + 1), size=12)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])
    fig.set_size_inches(6, 6)
    plt.show()

def convert_timestamp(date_str):
    if len(str(date_str).split(".")) > 2:
        timestamp = datetime.strptime(date_str, "%Y.%m.%d. %H:%M:%S.%f")
        return timestamp.timestamp()
    else:
        date_time = datetime.fromtimestamp(float(date_str))
        return date_time.strftime("%Y.%m.%d. %H:%M:%S.%f")


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def plot_mesh_permarray(mesh_obj, perm_array, ax=None, title="Mesh", sample_index=None):
    el_pos = np.arange(mesh_obj.n_el)
    pts = mesh_obj.node
    tri = mesh_obj.element
    x, y = pts[:, 0], pts[:, 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.tripcolor(x, y, tri, perm_array, shading="flat", edgecolor="k", alpha=0.8)

    ax.set_aspect("equal")
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])

    if sample_index is not None:
        ax.set_title(f"{title} Sample {sample_index}")
    else:
        ax.set_title(title)
    if ax is None:
        fig.colorbar(im, ax=ax)
        plt.show()
    else:
        plt.colorbar(im, ax=ax)


def seq_data(eit, perm, n_seg=4):
    sequence = [eit[i : i + n_seg] for i in range(len(eit) - n_seg)]
    aligned_perm = perm[n_seg:]
    return np.array(sequence), np.array(aligned_perm)

def plot_tank(r, h, ax):
    
    theta = np.linspace(0, 2 * np.pi, 100)
    z_cylinder = np.linspace(0, h, 100)
    X_cylinder, Z_cylinder = np.meshgrid(r * np.cos(theta), z_cylinder)
    Y_cylinder, _ = np.meshgrid(r * np.sin(theta), z_cylinder)
    ax.plot_surface(X_cylinder, Y_cylinder, Z_cylinder, color='lightgray', alpha=0.5)


def interpolate_equidistant_points(x, y, num_points):
    
    dx = np.diff(x)
    dy = np.diff(y)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    
    target_lengths = np.linspace(0, cumulative_lengths[-1], num_points)
    
    interp_x = interp1d(cumulative_lengths, x, kind='linear')
    interp_y = interp1d(cumulative_lengths, y, kind='linear')
    
    return interp_x(target_lengths), interp_y(target_lengths)

def create_trajectory(traj_type, radius, num_points, base_rotations=1):
    
    t = np.linspace(0, 2*np.pi, num_points)
    
    if traj_type == "circle":
        x_uniform = radius * np.cos(t)
        y_uniform = radius * np.sin(t)

    elif traj_type == "eight":
        x = radius * np.sin(t)
        y = radius * np.sin(2*t) / 2
        x_uniform, y_uniform = interpolate_equidistant_points(x, y, num_points)

    elif traj_type == "spiral":
        t = np.linspace(0, 2*np.pi*base_rotations, 1000)
        b = radius / (2*np.pi*base_rotations)
        r = b * t
        x = r * np.cos(t)
        y = -r * np.sin(t)
        x = x[::-1]
        y = y[::-1]
        x_uniform, y_uniform = interpolate_equidistant_points(x, y, num_points)

    elif traj_type == "polynomial":
        t = np.linspace(-2, 2, 1000)  
        x = t
        y = 0.5 * t * (t**2 - 1) * (t**2 - 4)
        max_val = np.max(np.abs([x, y]))
        scale = radius / max_val
        x *= scale
        y *= scale
        x_uniform, y_uniform = interpolate_equidistant_points(x, y, num_points)
        
    elif traj_type == "square":
      
        points_per_side = int(np.ceil(num_points / 4))

        corners = np.array([[radius, radius],[-radius, radius],[-radius, -radius],[radius, -radius],[radius, radius]])
        
        x = []
        y = []
      
        for i in range(4):
            
            side_x = np.linspace(corners[i][0], corners[i+1][0], points_per_side)
            side_y = np.linspace(corners[i][1], corners[i+1][1], points_per_side)
            

            if i < 3:
                x.extend(side_x[:-1])
                y.extend(side_y[:-1])
            else:
                x.extend(side_x)
                y.extend(side_y)
        
        x_array = np.array(x)
        y_array = np.array(y)
        x_uniform, y_uniform = interpolate_equidistant_points(x_array, y_array, num_points)
            
    return np.column_stack((x_uniform, y_uniform))

def create_trajectory_3D(traj_type, radius, num_points, base_rotations=10):

    if traj_type == "helix":
        t = np.linspace(0, 2*np.pi*base_rotations, num_points)
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = np.linspace(50, 100, num_points)

    elif traj_type == "spiral_helix":
        t = np.linspace(0, 2*np.pi*base_rotations, num_points) 
        
        b = radius / (2*np.pi*base_rotations)
        radius = radius - b * t
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = np.linspace(50, 100, num_points)

    elif traj_type == "circ_sin_wave":
        t = np.linspace(0, 2*np.pi*base_rotations, num_points)
        
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        
        z_amplitude = 25 
        z_frequency = 8  
        z_offset = 75    
        
        z = z_offset + z_amplitude * np.sin(t * z_frequency)
        
    return np.column_stack((x, y, z))


def load_sim_data(data_path, data_set):
    data_dirs = sorted(glob(f"{data_path}/{data_set}/"))  

    for i, directory in enumerate(data_dirs):
        file_list = sorted(glob(f"{directory}*.npz"))  
        voltage_list = []
        gamma_list = []
        anomaly_list = []
        pos_list = []

        for file in file_list:
            tmp = np.load(file, allow_pickle=True)  
            voltage_list.append(tmp["v"])  
            gamma_list.append(tmp["gamma"])
            anomaly_list.append(tmp["anomaly"])
            pos_list.append(tmp["pos"])

        voltage_array = np.array(voltage_list) 
        anomaly_array = np.array(anomaly_list)
        gamma_array = np.array(gamma_list)
        pos_array = np.array(pos_list)
    
    return voltage_array, gamma_array, pos_array 


def load_exp_data(data_path, data_set):
    
    data_dirs = sorted(glob(f"{data_path}/{data_set}/"))
        
    voltage_array = None
    temp_array = None
    timestamp_array = None
    position_array = None
    
    for i, directory in enumerate(data_dirs):
        file_list = sorted(glob(f"{directory}*.npz"))
            
        voltage_list = []
        temp_list = []
        timestamp_list = []
        position_list = []
     
        for file in file_list:
            tmp = np.load(file, allow_pickle=True)  
            voltage_list.append(tmp["v"])
            temp_list.append(tmp["temperature"])
            timestamp_list.append(tmp["timestamp"])
            position_list.append(tmp["position"])
            
        voltage_array = np.array(voltage_list)
        temp_array = np.array(temp_list)
        timestamp_array = np.array(timestamp_list)
        position_array = np.array(position_list)
        
    return voltage_array, temp_array, timestamp_array, position_array


def transform_position(position):

    center_offset = position[0]
    position_centered = position - center_offset
    tank_radius = 97       
    position_scaled_centered = position_centered / tank_radius
    
    return position_scaled_centered[1:]

def create_anomaly(mesh_obj, x, y, r):

    anomaly = [PyEITAnomaly_Circle(center=[x, y], r=r, perm=0.9)]
    ms = mesh.set_perm(mesh_obj, anomaly=anomaly, background=0.1)
    
    return ms
    
def create_gamma(positions):
    
    gamma_list = []
    n_el = 32
    mesh_obj = mesh.create(n_el, h0=0.05)
    for pos in positions:
        x, y = pos
        
        ms = create_anomaly(mesh_obj, x, y, 0.15)
        gamma_list.append(ms.perm_array)
    
    return np.array(gamma_list)  

def load_2Ddata(file_path, file_name, data_type, use_mean= True):

    if data_type == "sim":
        print("Loading simulation data...")
        voltage, gamma, pos = load_sim_data(file_path, file_name)
        print("Normalizing data...")
        voltage_normalized = (voltage - np.mean(voltage)) / np.std(voltage)
        
        gamma = gamma.reshape(-1, 2840, 1)
        voltage_normalized = voltage_normalized.reshape(-1, 32, 32, 1)
        voltage_seq, gamma_seq = seq_data(voltage_normalized, gamma, n_seg=4)

        print("Data loading complete!")
    
        return voltage_seq, gamma_seq, pos
    
    else:
        
        print("Loading experimental data...")
        voltage, temp, timestamp, position = load_exp_data(file_path, file_name)

        print("Calculating absolute voltages...")
        voltage_abs = np.abs(voltage) 
    
        print("Normalizing data...")
        voltage_normalized = (voltage_abs - np.mean(voltage_abs)) / np.std(voltage_abs)

        print("Subtracting voltage measurements from empty tank...")
        voltage_diff = voltage_normalized - voltage_normalized[0]
        voltage_diff = voltage_diff[1:]  

        if use_mean:
            print("Calculating mean of voltage differences...")
            voltage_diff = np.mean(voltage_diff, axis=1)
    

        print("Transforming position data...")
        trans_pos = transform_position(position)
        print(trans_pos.shape)
         
        print("Creating gamma...")
        gamma = create_gamma(trans_pos)
        print(gamma.shape)
        gamma = gamma.reshape(-1, 2840,1)
        voltage_diff = voltage_diff.reshape(-1, 32, 32, 1)
    
        if use_mean:
            print("Processing sequences with mean...")
            voltage_seq, gamma_seq = seq_data(voltage_diff, gamma, n_seg=4)
        else:
            print("Processing sequences for each measurement...")
            voltage_sequences = []
            gamma_sequences = []
        
            for idx in tqdm(range(5), desc="Processing measurement sequences"):
                v_seq, g_seq = seq_data(voltage_diff[:, idx, :, :], gamma, n_seg=4)
                voltage_sequences.append(v_seq)
                gamma_sequences.append(g_seq)
            
            print("Concatenating sequences...")
            voltage_seq = np.concatenate(voltage_sequences, axis=0)
            gamma_seq = np.concatenate(gamma_sequences, axis=0)
    
        print("Data loading complete!")
        return voltage_seq, gamma_seq, trans_pos, temp, timestamp

def transform_position_3D(position):
    
    center_offset = np.array([position[0,0], position[0,1], 0]) 
    position_centered = position - center_offset
    scale_factor = 32/194 
    scale_z = 24/140 
    
    position_centered[:,0] = position_centered[:,0] * scale_factor
    position_centered[:,1] = position_centered[:,1] * scale_z
    position_centered[:,2] = position_centered[:,2] * scale_z

    position_centered[:,0] = position_centered[:,0] + 16
    position_centered[:,1] = position_centered[:,1] + 16
    position_centered[:,2] = position_centered[:,2] + 8
    
    return position_centered[1:]

def create_gamma_3D(position):
    boundary = Boundary()
    gamma = list()
    diameter_labels = list()
    pos_ball = list()
    vol_ball = list()

    perm = 1
    d = 7

    for i, (x, y, z) in enumerate(position):
        ball = BallAnomaly(x, y, z, d, perm) 
        vxl_ball = voxel_ball(ball, boundary)
                
        gamma.append(vxl_ball)
        diameter_labels.append(d)  
        pos_ball.append([ball.y, ball.x, ball.z])
        vol_ball.append(np.where(vxl_ball == 1)[0].shape[0])

    gamma = np.array(gamma) / 2
    diameter_labels = np.array(diameter_labels)
    pos_ball = np.array(pos_ball)
    vol_ball = np.array(vol_ball)

    return gamma, diameter_labels, pos_ball, vol_ball

def load_3Ddata(file_path, file_name, use_mean= False):
    
    print("Loading experimental data...")
    voltage, temp, timestamp, position = load_exp_data(file_path, file_name)
    
    voltage_abs = np.abs(voltage) 
    
    voltage_normalized = (voltage_abs - np.mean(voltage_abs)) / np.std(voltage_abs)

    voltage_diff = voltage_normalized - voltage_normalized[0]
    voltage_diff = voltage_diff[1:] 

    if use_mean:
        print("Calculating mean of voltage differences...")
        voltage_diff = np.mean(voltage_diff, axis=1)
    
    print("Transforming position data...")
    trans_pos = transform_position_3D(position)
    
    print("Creating gamma...")
    gamma, _, _, _ = create_gamma_3D(trans_pos)
    
    if use_mean:
        print("Processing sequences with mean...")
        voltage_seq, gamma_seq = seq_data(voltage_diff, gamma, n_seg=4)
    else:
        print("Processing sequences for each measurement...")
        voltage_sequences = []
        gamma_sequences = []
        
        for idx in tqdm(range(5), desc="Processing measurement sequences"):
            v_seq, g_seq = seq_data(voltage_diff[:, idx, :, :], gamma, n_seg=4)
            voltage_sequences.append(v_seq)
            gamma_sequences.append(g_seq)
            
        print("Concatenating sequences...")
        voltage_seq = np.concatenate(voltage_sequences, axis=0)
        gamma_seq = np.concatenate(gamma_sequences, axis=0)
    
    print("Data loading complete!")
    return voltage_seq, gamma_seq, trans_pos, temp, timestamp


def voxel_ball(ball, boundary, empty_gnd=0, mask=False):
    y, x, z = np.indices((boundary.x_length, boundary.y_length, boundary.z_length))
    voxel = (
        np.sqrt((x - ball.x) ** 2 + (y - ball.y) ** 2 + (z - ball.z) ** 2) < ball.d / 2
    )
    if mask:
        return voxel
    else:
        return np.where(voxel, ball.perm, empty_gnd)

def voxel_ball(ball, boundary, empty_gnd=0, mask=False):
    
    #scale_factor = 32 / 194  # 32 voxels / 194mm (tank diameter)
    #ball_diameter_voxels = round(ball.d * scale_factor)
    #print(f"ball_diameter_voxels: {ball_diameter_voxels}")
    
    y, x, z = np.indices((boundary.x_length, boundary.y_length, boundary.z_length))
    voxel = (
        np.sqrt((x - ball.x) ** 2 + (y - ball.y) ** 2 + (z - ball.z) ** 2) < ball.d / 2
    )
    if mask:
        return voxel
    else:
        return np.where(voxel, ball.perm, empty_gnd)

def create_cylinder_mesh(tank, n_points=100):
    
    theta = np.linspace(0, 2*np.pi, n_points)
    z = np.linspace(tank.T_bz[0], tank.T_bz[1], n_points)
    theta, z = np.meshgrid(theta, z)
    x = tank.T_r * np.cos(theta)
    y = tank.T_r * np.sin(theta)
    return x, y, z

def plot_tank_and_ball(ball, tank, boundary):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
   
    scale_factor = 32 / tank.T_d 
    
    cylinder_radius = tank.T_r * scale_factor
    cylinder_height = tank.T_bz[1] * scale_factor
    x_cyl, y_cyl, z_cyl = create_cylinder_mesh(cylinder_radius, cylinder_height)
    
    x_cyl += 16  
    y_cyl += 16  
    
    ax.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.1, color='gray')
    
    ball_voxels = voxel_ball(ball, boundary)
    
    ax.voxels(ball_voxels.transpose(1, 0, 2), facecolors='cornflowerblue', alpha=0.8)
    
    ax.plot([0, boundary.x_length], [0, 0], [0, 0], 'k-', linewidth=1)
    #ax.text(boundary.x_length+1, 0, 0, 'x')
    
    ax.plot([0, 0], [0, boundary.y_length], [0, 0], 'k-', linewidth=1)
    #ax.text(0, boundary.y_length+1, 0, 'y')
    
    ax.plot([0, 0], [0, 0], [0, boundary.z_length], 'k-', linewidth=1)
    #ax.text(0, 0, boundary.z_length+1, 'z')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.set_xlim([boundary.x_0, boundary.x_length])
    ax.set_ylim([boundary.y_0, boundary.y_length])
    ax.set_zlim([boundary.z_0, boundary.z_length])
    
    # Set grid with minor lines
    ax.grid(True)
    
    # Set major ticks to show 0,8,16,24,32
    ax.set_xticks([0, 8, 16, 24, 32])
    ax.set_yticks([0, 8, 16, 24, 32])
    ax.set_zticks([0, 8, 16, 24, 32])
    
    # Set view angle
    ax.view_init(elev=10, azim=45)
    
    plt.tight_layout()
    plt.show()

def plot_ball(
    ball: BallAnomaly,
    boundary: Boundary,
    res: int = 50,
    elev: int = 25,
    azim: int = 10,
):
    u = np.linspace(0, 2 * np.pi, res)
    v = np.linspace(0, np.pi, res)

    x_c = ball.x + ball.d / 2 * np.outer(np.cos(u), np.sin(v))
    y_c = ball.y + ball.d / 2 * np.outer(np.sin(u), np.sin(v))
    z_c = ball.z + ball.d / 2 * np.outer(np.ones(np.size(u)), np.cos(v))

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    # ball
    ax.plot_surface(x_c, y_c, z_c, color="C0", alpha=1)

    ax.set_xlim([boundary.x_0, boundary.x_length])
    ax.set_ylim([boundary.y_0, boundary.y_length])
    ax.set_zlim([boundary.z_0, boundary.z_length])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()


def plot_voxel_c(voxelarray, elev=20, azim=10):

    ax = plt.figure(figsize=(4, 4)).add_subplot(projection="3d")
    ax.voxels(
        voxelarray.transpose(1, 0, 2), facecolors=colors[int(np.max(voxelarray) - 1)]
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(azim=azim, elev=elev)
    plt.tight_layout()
    plt.show()

def plot_voxel(voxelarray, fc=0, elev=20, azim=10):
    ax = plt.figure(figsize=(4, 4)).add_subplot(projection="3d")
    ax.voxels(voxelarray.transpose(1, 0, 2), facecolors=f"C{fc}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(azim=azim, elev=elev)
    plt.tight_layout()
    plt.show()

def plot_temperature_vs_time(time, temp, fpath ='', fname = '', save = False):

    ensure_dir(fpath)
    fig, ax = plt.subplots(figsize=(10, 6))
    xlabel="Zeit"
    ylabel="Temperatur (°C)"

    ax.plot(time, temp, linestyle='-', color='steelblue', linewidth=1)
  
    n = len(time)
    tick_indices = [
        0,                  
        n // 4,            
        n // 2,            
        3 * n // 4,        
        n - 1              
    ]
    
    dt = datetime.strptime(time[0], "%Y-%m-%d %H:%M:%S")
    date_str = dt.strftime("%d.%m.%Y")
   
    def format_time_with_index(ts, idx):
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        time_str = dt.strftime("%H:%M")
        return f"{time_str}\n({idx})"
    
    ax.set_xticks([time[i] for i in tick_indices])
    ax.set_xticklabels([format_time_with_index(time[i], i) for i in tick_indices])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f"{os.path.join(fpath, fname)}.pdf", format = 'pdf', bbox_inches = 'tight')
        print(f"Plot saved to {os.path.join(fpath, fname)}.pdf")