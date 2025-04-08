from glob import glob
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from pyeit import mesh
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from PIL import Image  
from mpl_toolkits.mplot3d import Axes3D
from src.util import ensure_dir

def compute_evaluation_metrics(mesh_obj, true_perm, pred_perm, threshold=0.5):
   
    tri_centers = np.mean(mesh_obj.node[mesh_obj.element], axis=1)
   
    true_indices = np.where(true_perm > threshold)[0]
    pred_indices = np.where(pred_perm > threshold)[0]
    
    
    true_coords = tri_centers[true_indices]
    pred_coords = tri_centers[pred_indices]
    
    true_center = np.mean(true_coords, axis=0)
    pred_center = np.mean(pred_coords, axis=0)
    
    true_center = np.round(true_center, 3)
    pred_center = np.round(pred_center, 3)
    
    delta_x = pred_center[0] - true_center[0]
    delta_y = pred_center[1] - true_center[1]
    delta_elements = len(pred_indices)-len(true_indices) 
    
    deviation_metrics = (delta_x, delta_y, delta_elements)
    coordinates = (true_center, pred_center, true_coords, pred_coords)
    
    return deviation_metrics, coordinates

def plot_deviations_x_y(true_perms, pred_perms, mesh_obj, threshold=0.5, save=False, 
                       fpath='', fname='', limits=(-1.0, 1.0), scale_factor=True, figsize=(8, 8), tick_fontsize=18, label_fontsize=25):
    x_deviations = []
    y_deviations = []
    
    valid_metrics = 0
    invalid_metrics = 0
    
    for true_perm, pred_perm in zip(true_perms, pred_perms):
        metrics, *_ = compute_evaluation_metrics(mesh_obj, true_perm, pred_perm, threshold)
        if metrics is not None:
            delta_x, delta_y, _ = metrics
            x_deviations.append(delta_x) 
            y_deviations.append(delta_y)
            valid_metrics += 1
        else:
            invalid_metrics += 1
    
    if invalid_metrics > 0:
        print(f"Warning: {invalid_metrics} of {valid_metrics + invalid_metrics} data points could not be calculated.")
    
    x_deviations = np.array(x_deviations)
    y_deviations = np.array(y_deviations)
    
    valid_indices = ~(np.isnan(x_deviations) | np.isnan(y_deviations))
    if not np.all(valid_indices):
        print(f"Warning: {np.sum(~valid_indices)} of {len(valid_indices)} data points contain NaN values and are removed.")
        x_deviations = x_deviations[valid_indices]
        y_deviations = y_deviations[valid_indices]

    
    if scale_factor:
        scaling = 97 
        x_deviations = x_deviations * scaling
        y_deviations = y_deviations * scaling
        min_limit, max_limit = limits
        limits = (min_limit * 100, max_limit * 100)
 
        
    df = pd.DataFrame({
        'x-deviation': x_deviations,
        'y-deviation': y_deviations
    })
    
    plt.rcParams.update({'font.family': 'serif'})
    sns.set_style("whitegrid", {'axes.facecolor': '#eaeaf2',
                               'grid.color': 'white',
                               'grid.linestyle': '-'})
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': tick_fontsize, 
        'axes.labelsize': label_fontsize,  
        'xtick.labelsize': tick_fontsize,  
        'ytick.labelsize': tick_fontsize   
    })
    
    g = sns.jointplot(
        data=df,
        x='x-deviation',
        y='y-deviation',
        kind='kde',
        xlim=limits,
        ylim=limits,
        height=figsize[0]
    )
    
    g.ax_joint.set_facecolor('#eaeaf2')
    g.ax_marg_x.set_facecolor('#eaeaf2')
    g.ax_marg_y.set_facecolor('#eaeaf2')
    
    g.plot_joint(sns.kdeplot, fill=True, levels=10, cmap='viridis')
    sns.kdeplot(data=df, x='x-deviation', ax=g.ax_marg_x, color='steelblue', fill=True)
    sns.kdeplot(data=df, y='y-deviation', ax=g.ax_marg_y, color='steelblue', fill=True)
    
    x_mean = np.mean(x_deviations)
    y_mean = np.mean(y_deviations)
    g.ax_joint.axvline(x=x_mean, color='r', linestyle='--', alpha=0.5, linewidth=2)
    g.ax_joint.axhline(y=y_mean, color='r', linestyle='--', alpha=0.5, linewidth=2)
    
    num_ticks = 5  
    ticks = np.linspace(limits[0], limits[1], num_ticks)
    ticks = np.round(ticks, 2)
    
    g.ax_joint.set_xticks(ticks)
    g.ax_joint.set_yticks(ticks)
    
    g.ax_joint.tick_params(axis='both', labelsize=tick_fontsize)
    g.ax_marg_x.tick_params(labelsize=tick_fontsize)
    g.ax_marg_y.tick_params(labelsize=tick_fontsize)
    
    unit_suffix = " [mm]" if scale_factor else ""
    g.ax_joint.set_xlabel(f'x-deviation{unit_suffix}', labelpad=10, fontsize=label_fontsize)
    g.ax_joint.set_ylabel(f'y-deviation{unit_suffix}', labelpad=10, fontsize=label_fontsize)
    
    x_std = np.std(x_deviations)
    y_std = np.std(y_deviations)
    
    if save and fpath and fname:
        os.makedirs(fpath, exist_ok=True)
        stats_filename = os.path.join(fpath, fname.replace('.pdf', '') + '_stats.txt')
        with open(stats_filename, 'w') as f:
            f.write(f'μx = {x_mean:.3f}\n')
            f.write(f'μy = {y_mean:.3f}\n')
            f.write(f'σx = {x_std:.3f}\n')
            f.write(f'σy = {y_std:.3f}\n')
            f.write(f'valid data points: {len(x_deviations)}\n')
            if invalid_metrics > 0:
                f.write(f'invalid data points: {invalid_metrics}\n')
        print(f"satistics stored in: {stats_filename}")
    else:
        
        print(f'μx = {x_mean:.3f}')
        print(f'μy = {y_mean:.3f}')
        print(f'σx = {x_std:.3f}')
        print(f'σy = {y_std:.3f}')
        print(f'valid data points: {len(x_deviations)}')
        if invalid_metrics > 0:
            print(f'invalid data points: {invalid_metrics}')
    
    if save and fpath and fname:
        os.makedirs(fpath, exist_ok=True)
        
        png_filename = fname.replace('.pdf', '.png')
        png_path = os.path.join(fpath, png_filename)
        plt.figure(g.fig.number)
        plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
        print(f"PNG saved in: {png_path}")
        
        pdf_filename = fname if fname.endswith('.pdf') else fname + '.pdf'
        pdf_path = os.path.join(fpath, pdf_filename)
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"PDF saved in: {pdf_path}")
        
        os.remove(png_path)
        print(f"temporary PNG file removed: {png_path}")
    
    return g
    
    
def plot_deviations_perm(true_perms, pred_perms, mesh_obj, threshold=0.3, 
                        save=False, fpath='', fname='', binwidth=10):
    
    perm_deviations = []
    
    for true_perm, pred_perm in zip(true_perms, pred_perms):
        metrics, _ = compute_evaluation_metrics(mesh_obj, true_perm, pred_perm, threshold)
        if metrics is not None:
            _, _, delta_perm = metrics
            perm_deviations.append(delta_perm)
    
    plt.rcParams.update({'font.family': 'serif'})
    sns.set_style("whitegrid", {'axes.facecolor': '#eaeaf2',
                               'grid.color': 'white',
                               'grid.linestyle': '-'})
    
    plt.figure(figsize=(7, 7))
    plt.autoscale()
    
    p = sns.histplot(data=perm_deviations,
                    binwidth=binwidth,
                    kde=True,
                    color='steelblue',  
                    alpha=0.6,
                    kde_kws={'bw_adjust': 1.2,  
                            'cut': 3,           
                            'gridsize': 200})   
    
    p.set_facecolor('#eaeaf2')
    
    mean_dev = np.mean(perm_deviations)
    std_dev = np.std(perm_deviations)
    st_fe = 2840 
    percent_dev = (mean_dev/st_fe) * 100
    
    p.axvline(x=mean_dev, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    p.set_xlabel("Deviating elements", fontsize=20)
    p.set_ylabel("Number", fontsize=20)
    
    p.tick_params(axis='both', labelsize=15)
    
    p.grid(True, alpha=0.5)
    
    for spine in p.spines.values():
        spine.set_visible(True)
    
    fig = p.get_figure()
    
    if save:
        os.makedirs(fpath, exist_ok=True)
        
        plt.tight_layout()
        
        pdf_path = os.path.join(fpath, fname + '.pdf')
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"PDF saved to: {pdf_path}")
        
        stats_filename = os.path.join(fpath, fname + '_stats.txt')
        with open(stats_filename, 'w') as f:
            f.write(f'mean perm deviation: {round(mean_dev)} [FE]\n')
            f.write(f'standard deviation: {round(std_dev, 1)} [FE]\n')
            #f.write(f'Prozentuale Abweichung: {round(percent_dev, 2)} [%]\n')
        print(f"statistics saved to: {stats_filename}")



def plot_seq_recon_examples(mesh_obj, true_perm, pred_perm, pos, 
                          save=False, fpath='', fname='', threshold = 0.3,perm_0=0, num_plots=25):
    
    seq_length = len(true_perm)
    first_idx = 0
    last_idx = seq_length - 1
    middle_indices = np.linspace(first_idx, last_idx, num_plots, dtype=int)
    sequential_indices = middle_indices.tolist()
    
    cols = 5
    rows = int(np.ceil(num_plots / cols))  
    fig = plt.figure(figsize=(4*cols, 4*rows), facecolor='white')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    pts = mesh_obj.node
    tri = mesh_obj.element
    x, y = pts[:, 0], pts[:, 1]
    tri_centers = np.mean(pts[tri], axis=1)
    
    #true_centers = []
    #for idx in range(len(true_perm)):
       # mask = true_perm[idx].flatten() > threshold
       # if np.any(mask):
       #     centers = tri_centers[mask]
        #    true_centers.append(np.mean(centers, axis=0))
    #true_centers = np.array(true_centers)

    true_centers = pos
    
    for i, idx in enumerate(sequential_indices):
            
        true_values = true_perm[idx].flatten()
        pred_values = pred_perm[idx].flatten()
        ax = plt.subplot(rows, cols, i + 1)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('white')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.5)
        
       
        ax.set_title(f't = {idx+4}', fontsize=20, fontweight='bold', pad=8)
        
        im_pred = ax.tripcolor(x, y, tri, pred_values, shading="flat", 
                             edgecolor='none', linewidth=0.001, 
                             alpha=1, cmap='viridis',
                             vmin=0, vmax=1)
        
        mask = true_values > threshold
       
        for j in np.where(mask)[0]:
            triangle = tri[j]
            triangle_pts = pts[triangle]
            ax.fill(triangle_pts[:, 0], triangle_pts[:, 1], 
                    color='darkorange', alpha=0.6,
                    edgecolor='none', linewidth=0.001)
    
        ax.plot(pos[:, 0], pos[:, 1], 
                color='yellow', linestyle='--', linewidth=1.5)
        
        ax.set_ylim([-1, 1])
        ax.set_xlim([-1, 1])
        ax.set_aspect('equal', adjustable='box', anchor='C')
    
    plt.tight_layout()
    if save:
        base_path = os.path.splitext(fpath + fname)[0]
        png_path = base_path + '.png'
        pdf_path = base_path + '.pdf'
      
        plt.savefig(png_path, bbox_inches='tight', dpi=300)
        image = Image.open(png_path)
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        image.save(pdf_path, 'PDF', resolution=300.0)
        print(f"Plots saved as:\nPNG: {png_path}\nPDF: {pdf_path}")
        os.remove(png_path)
    plt.show()


### Evaluation for 3D Reconstruction!!
from src.classes import Boundary, TankProperties32x2, BallAnomaly

def create_cylinder_mesh(n_points=30): 
    tank = TankProperties32x2()
    scale_factor = 32 / tank.T_d 
    cylinder_radius = tank.T_r * scale_factor
    cylinder_height = tank.T_bz[1] * scale_factor

    theta = np.linspace(0, 2*np.pi, n_points)
    z = np.linspace(0, cylinder_height, n_points)
    theta, z = np.meshgrid(theta, z)
    x = cylinder_radius * np.cos(theta)
    y = cylinder_radius * np.sin(theta)
    x += 16  
    y += 16  

    return x, y, z

def plot_voxel_data(data, pos, indices, x_cyl, y_cyl, z_cyl, colors, 
                   n_per_row, n_rows, elev, azim, figsize=(16, 12)):

    pos_line_x = pos[:, 0]
    pos_line_y = pos[:, 1]
    pos_line_z = pos[:, 2]
    
    fig = plt.figure(figsize=figsize)
    
    for idx in range(min(len(indices), n_rows * n_per_row)):
        ax = fig.add_subplot(n_rows, n_per_row, idx + 1, projection='3d')
        
        ax.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.15, color='gray')
    
        ax.voxels(data[idx].transpose(1, 0, 2),
                 facecolors=colors[int(np.max(data[idx]) - 1)])
        
        # ax.plot(pos_line_y, pos_line_x, pos_line_z, 'k-', linewidth=1, alpha=0.7)
        
        ax.set_xticks(np.linspace(0, 32, 5))
        ax.set_yticks(np.linspace(0, 32, 5))
        ax.set_zticks(np.linspace(0, 32, 5))
        
        ax.set_xticklabels(['0','', '', '', '32'], fontsize=12)
        ax.set_yticklabels(['0','', '', '', '32'], fontsize=12)
        ax.set_zticklabels(['0','', '', '', '32'], fontsize=12)
        
        ax.grid(True, alpha=0.3)
        
        ax.set_xlabel('y', fontsize=14, labelpad=-10)
        ax.set_ylabel('x', fontsize=14, labelpad=-10)
        ax.set_zlabel('z', fontsize=14, labelpad=-10, rotation=0)
        
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 32)
        ax.set_zlim(0, 32)
        ax.view_init(azim=azim, elev=elev)
        
        ax.set_title(f"t = {indices[idx]+4}", fontweight='bold', fontsize=14, pad=0)
    
    plt.tight_layout(h_pad=-1.5, w_pad=-5)
    return fig

def center_of_mass(voxel_matrix):
    
    total_mass = np.sum(voxel_matrix)
    if total_mass == 0:
        return None
        
    x_coords, y_coords, z_coords = np.meshgrid(
        np.arange(voxel_matrix.shape[0]),
        np.arange(voxel_matrix.shape[1]),
        np.arange(voxel_matrix.shape[2]),
    )
    
    center_x = np.sum(x_coords * voxel_matrix) / total_mass
    center_y = np.sum(y_coords * voxel_matrix) / total_mass
    center_z = np.sum(z_coords * voxel_matrix) / total_mass

    return np.array([center_y, center_x, center_z])

def compute_pos_error(predicted_voxels, true_voxels):
    pos_err = list()
    zero_matrices = 0
    
    for i, (pred, true) in enumerate(zip(predicted_voxels, true_voxels)):
        
        com_pred = center_of_mass(pred)
        com_true = center_of_mass(true)
        
        if com_pred is None or com_true is None:
            zero_matrices += 1
            continue  

        pos_err.append(com_true - com_pred)
    
    pos_err = np.array(pos_err)
    print(f"number of ignored matrices: {zero_matrices}")
    print(f"number of pairs taken into account {len(pos_err)}")
    
    return pos_err

def compute_volume_error(predicted_voxels, true_voxels):
        v_err = list()
        for pred, true in zip(predicted_voxels, true_voxels):
            ele_pred = len(np.where(pred != 0)[0])
            ele_true = len(np.where(true != 0)[0])
            if ele_pred == 0:
                v_err.append(None)
            else:
                v_err.append(ele_pred - ele_true)
        v_err = np.array(v_err)
        return v_err

def plot_voxel_overlay(gamma_true, gamma_true_pred, pos, save = False, fpath = '', fname ='', elev = 30, azim = 20, true_color='royalblue', pred_color='darkorange'):           
    figsize=(16, 16)
    n_samples=16
    n_per_row=4
    pred_alpha=0.5

    ensure_dir(fpath)
    
    n_rows = int(np.ceil(n_samples / n_per_row))
    
    total_samples = gamma_true.shape[0]
    indices = np.linspace(0, total_samples-1, n_samples, dtype=int)
    
    true_samples = np.array([gamma_true[idx, ...] for idx in indices])
    pred_samples = np.array([gamma_true_pred[idx, ...] for idx in indices])
    
    x_cyl, y_cyl, z_cyl = create_cylinder_mesh()
    
    pos_line_x = pos[:, 0]
    pos_line_y = pos[:, 1]
    pos_line_z = pos[:, 2]
    
    fig = plt.figure(figsize=figsize)
    
    for idx in range(min(len(indices), n_rows * n_per_row)):
        ax = fig.add_subplot(n_rows, n_per_row, idx + 1, projection='3d')
        
        ax.plot_surface(x_cyl, y_cyl, z_cyl, alpha=0.15, color='gray')
        
        true_voxels = true_samples[idx].transpose(1, 0, 2)
        true_mask = true_voxels > 0
        ax.voxels(true_mask, facecolors=true_color, alpha=1.0)
        
        pred_voxels = pred_samples[idx].transpose(1, 0, 2)
        pred_mask = pred_voxels > 0
        ax.voxels(pred_mask, facecolors=pred_color, alpha=pred_alpha)
        
        ax.set_xticks(np.linspace(0, 32, 5))
        ax.set_yticks(np.linspace(0, 32, 5))
        ax.set_zticks(np.linspace(0, 32, 5))
        
        ax.set_xticklabels(['0','', '', '', '32'], fontsize=12)
        ax.set_yticklabels(['0','', '', '', '32'], fontsize=12)
        ax.set_zticklabels(['0','', '', '', '32'], fontsize=12)
        
        ax.grid(True, alpha=0.3)
        
        ax.set_xlabel('y', fontsize=14, labelpad=-10)
        ax.set_ylabel('x', fontsize=14, labelpad=-10)
        ax.set_zlabel('z', fontsize=14, labelpad=-10, rotation=0)
        
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 32)
        ax.set_zlim(0, 32)
        ax.view_init(azim=azim, elev=elev)
        
        ax.set_title(f"t = {indices[idx]+4}", fontweight='bold', fontsize=14, pad=0)
    
    
    plt.tight_layout(h_pad=-1.5, w_pad=3, rect=[0, 0.05, 1, 1])
    if save:
        plt.savefig(f"{os.path.join(fpath, fname)}.pdf", format='pdf', bbox_inches='tight')
        print(f"Overlay plot saved to {os.path.join(fpath, fname)}.pdf")


def plot_volume_deviations(predicted_voxels, true_voxels, 
                        fpath='',
                        fname='',
                        save = False,
                        xlabel_size=20,
                        ylabel_size=20,
                        tick_size=15,
                        color='steelblue',
                        kde=True,
                        kde_bw_adjust=1.2,
                        kde_cut=3,
                        kde_gridsize=200,
                        show_mean_line=True,
                        mean_line_color='red'):

    
    binwidth=50
    figsize=(7, 7)
    font_family='serif'
    grid_alpha=0.5
    
    ensure_dir(fpath)
    
    v_err = compute_volume_error(predicted_voxels, true_voxels)
    
    v_err = v_err[v_err != None]  
    
    v_err = v_err.astype(float)
    
    plt.rcParams.update({'font.family': font_family})
    sns.set_style("whitegrid", {'axes.facecolor': '#eaeaf2',
                               'grid.color': 'white',
                               'grid.linestyle': '-'})
    
    plt.figure(figsize=figsize)
    plt.autoscale()
    
    p = sns.histplot(data=v_err, 
                    binwidth=binwidth,  
                    kde=kde,
                    color=color,
                    alpha=0.6,
                    kde_kws={'bw_adjust': kde_bw_adjust,
                            'cut': kde_cut,
                            'gridsize': kde_gridsize})
    
    p.set_facecolor('#eaeaf2')
    
    mean_dev = np.mean(v_err)
    std_dev = np.std(v_err)
    
    if show_mean_line:
        p.axvline(x=mean_dev, color=mean_line_color, linestyle='--', alpha=0.5, linewidth=2)
    
    p.set_xlabel("deviating voxels", fontsize=xlabel_size)
    p.set_ylabel("number", fontsize=ylabel_size)
    
    p.tick_params(axis='both', labelsize=tick_size)
    
    p.grid(True, alpha=grid_alpha)
    
    for spine in p.spines.values():
        spine.set_visible(True)
    
    plt.tight_layout()
    
    fig = p.get_figure()

    if save:
        plt.tight_layout()
        
        pdf_path = os.path.join(fpath, f"{fname}_volume_deviations.pdf")
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"PDF saved to: {pdf_path}")
        
        stats_filename = os.path.join(fpath, f"{fname}_volume_deviations_stats.txt")
        with open(stats_filename, 'w') as f:
            f.write(f'mean volume deviation: {round(mean_dev)} Voxel\n')
            f.write(f'standard deviation: {round(std_dev, 1)} Voxel\n')
        print(f"Statistics saved to: {stats_filename}")
 
    plt.show()

    print(f'mean volume deviation: {round(mean_dev)} Voxel')
    print(f'standard deviation: {round(std_dev, 1)} Voxel')

def plot_position_error(gamma_true_pred, gamma_true, 
                    fpath='Abbildung_3D/',
                    fname='seq_recon',
                    save = False,
                    whisker_length=1.5, 
                    box_color='steelblue',
                    box_alpha=0.6,
                    axis_labels=['x-Achse', 'y-Achse', 'z-Achse'],
                    show_stats=True):
    scale_factor=194/32
    ensure_dir(fpath)
    figsize=(8, 8)
    pos_err = compute_pos_error(gamma_true_pred, gamma_true)
    
    pos_err_mm = pos_err * scale_factor
    
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'figure.edgecolor': 'white',
        'savefig.edgecolor': 'white'
    })
    
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_facecolor('white')
    
    bp = plt.boxplot([pos_err_mm[:, 0], pos_err_mm[:, 1], pos_err_mm[:, 2]], 
                    labels=axis_labels,
                    patch_artist=True,
                    whis=whisker_length,  
                    showfliers=True) 
    
    for patch in bp['boxes']:
        patch.set_facecolor(box_color)
        patch.set_alpha(box_alpha)
    
    for median in bp['medians']:
        median.set(color='red', linewidth=2)
    
    for flier in bp['fliers']:
        flier.set(marker='o', 
                 markerfacecolor='black', 
                 markeredgecolor='none',
                 markersize=3,
                 alpha=0.5)
    
    yticks = np.arange(-120, 121, 40)  
    
    plt.xlabel('', fontsize=20)  
    plt.ylabel('deviationg [mm]', fontsize=25)
    plt.xticks(fontsize=20)
    plt.xlabel('spatial axes', fontsize=25)
    plt.yticks(yticks, fontsize=20)
    
    plt.grid(False) 

    for y in [-80, -40, 0, 40, 80]:
        plt.axhline(y=y, color='gray', linestyle='--', alpha=0.7)
        
    ax.xaxis.grid(False)
    ax.set_axisbelow(True) 
    
    plt.ylim(-120, 120)
    
    plt.tight_layout()
    
    if save:
        save_path = os.path.join(fpath, f"{fname}_position_error.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    #if show_stats:
    #    print("\nStatistische Kennzahlen:")
    #    print(f"{'Achse':>10} {'Mittelwert':>12} {'Std.Abw.':>10} {'Median':>10} {'IQR':>10}")
    #    print("-" * 60)
        
    #    for i, axis in enumerate(axis_labels):
    #        data = pos_err_mm[:, i]
    #        mean = np.mean(data)
    #        std = np.std(data)
    #        median = np.median(data)
    #        q1 = np.percentile(data, 25)
    #        q3 = np.percentile(data, 75)
    #        iqr = q3 - q1
            
     #       print(f"{axis:>10} {mean:>12.3f} {std:>10.3f} {median:>10.3f} {iqr:>10.3f}")
      
     #   print("\nAusreißer nach 1.5 IQR Regel:")
     #   print(f"{'Achse':>10} {'Anzahl':>10} {'Prozent':>10}")
     #   print("-" * 35)
        
     #   for i, axis in enumerate(axis_labels):
     #       data = pos_err_mm[:, i]
     #       q1 = np.percentile(data, 25)
     #       q3 = np.percentile(data, 75)
     #       iqr = q3 - q1
            
     #       lower_bound = q1 - 1.5 * iqr
     #       upper_bound = q3 + 1.5 * iqr
            
     #       outliers = data[(data < lower_bound) | (data > upper_bound)]
     #       outlier_count = len(outliers)
     #       outlier_percent = outlier_count / len(data) * 100
            
     #       print(f"{axis:>10} {outlier_count:>10} {outlier_percent:>9.2f}%")