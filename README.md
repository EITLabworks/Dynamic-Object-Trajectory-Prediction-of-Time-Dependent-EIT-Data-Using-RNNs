# Dynamic-object-trajectory-prediction-of-time-dependent-EIT-data-using-recurrent-neural-networks

This project presents a novel approach for dynamic image reconstruction of Electrical Impedance Tomography (EIT). This approach uses a data-driven reconstruction model consisting of a Variational Autoencoder (VAE) and a mapper with an integrated Long-Short-Term-Memory (LSTM) unit. The network has been specically designed for dynamic object trajectory prediction, allowing accurate tracking of an object's movement within the EIT tank and also predicting future object positions by exploiting temporal information in sequential EIT data. This approach was developed for 2D and 3D reconstructions of object motion. Data collection was performed using FEM simulation (pyEIT forward solver) for simulation data and an EIT tank equipped with two electrode rings (32 electrodes each) and a Sciospec EIT device for experimental data. In this project, the reconstruction network was trained and tested on simulation data, experimental EIT data collected during 2D motion and experimental EIT data collected during 3D motion.

## Reconstruction network architecture

The reconstruction model consists of two core components: a mapper with an integrated LSTM layer at the output and a VAE decoder. The architecture is illustrated in figure 1.

<p align="center">
  
  <img src="images/reconstruction_model.png" alt="Empty_mesh" width="50%">

</p>
<p align="center" style="font-size: smaller;">
  <em>Figure 1: Architecture of reconstruction model.</em>
</p>

The LSTM mapper, denoted as $\Xi$, processes temporal sequences of voltage measurements and maps it to the latent space $\mathbf{h}$. Subsequently, the VAE decoder, denoted as $\Psi$, reconstructs the latent representation into a conductivity distribution. The complete reconstruction network $\Gamma$ is defined as the composition of these mapping processes:

$$
\Gamma := \Xi \circ \Psi : V_{t} \mapsto h_{t+1} \mapsto \hat{\gamma}_{t+1}
$$

Here, $V_{t}$ represents the voltage measurements at time $t$, $h_{t+1}$ the predicted latent space representation at time $t+1$, and $\hat{\gamma}_{t+1}$ is the reconstructed conductivity distribution at time $t+1$. Figure 2 illustrates the working principle of the reconstruction network, demonstration how a sequence of voltage measurements as input of the network is used to predict the future conductivity distribution.

<p align="center">
  <img src="images/reconstruction_process.png" width="50%">
</p>
<p align="center" style="font-size: smaller;">
  <em>Figure 2: Overview of the reconstruction process of the proposed reconstruction model. A sequence of four voltage measurements is used to predict the conductivity distribution of the next time step.</em>
</p>

## Training of reconstruction network

The training process was conducted in two stages. In the first stage, the VAE was trained in an unsupervised using synthetically generated conductivity distributions for both 2D and 3D space.
For the 2D reconstructions, a triangular mesh representing the electrode plane of a cylindrical tank was used. For 3D reconstructions, a voxel-based approach was used.
In the second training stage, the LSTM mapper was trained in a supervised manner. The VAE encoder generated a latent representations of known conductity distributions, which served as labels for the supervised learning of the LSTM mapper. Sequences of voltage measurements were paired with the corresponding latent representations of future conductivity distributions.

## EIT data collection

EIT data were acquired in both simulated and experimental settings. Simulations were performed using FEM-based modeling with the pyEIT package, while experimental data were collected using an EIT water tank. For 2D data, both FEM simulation and experimental measurements were conducted on a single electrode plane, yielding $32^2$ voltage data points per frame. For 3D data, experimental measurements with two electrode planes were performed, resulting in $64^2$ voltage data points per frame. The EIT data were collected by tracking an acrylic ball along predefined trajectories at discrete positions. In 2D space, a circular, spiral, eight, polynomial, square trajectory were used. In 3D space, the trajectories uses were a helix, a spiral helix and a circular sine wave.

# Results 

## 2D simulation model

The 2D simulation model was trained on a spiral trajectory and tested on circular and eight shaped trajectory. The results demonstrate high predicition accuracy for the proposed resonstruction network.

<table width="1000px" style="table-layout: fixed; background-color:#1a1a1a; color:white; border-collapse:collapse; margin-bottom:30px;">
  <tr>
    <th width="500px" style="text-align:center; padding:10px 0; border:1px solid #333; background-color:#1a1a1a; font-weight:bold;">
      Circle Trajectory
    </th>
    <th width="500px" style="text-align:center; padding:10px 0; border:1px solid #333; background-color:#1a1a1a; font-weight:bold;">
      Eight Trajectory
    </th>
  </tr>
  <tr>
    <td width="500px" height="300px" style="text-align:center; vertical-align:middle; padding:15px; border:1px solid #333;">
      <div style="height:280px; display:flex; align-items:center; justify-content:center;">
        <img src="results/2D reconstruction/sim reconstruction/circle_recon.gif" style="max-width:280px; max-height:280px; object-fit:contain;">
      </div>
    </td>
    <td width="500px" height="300px" style="text-align:center; vertical-align:middle; padding:15px; border:1px solid #333;">
      <div style="height:280px; display:flex; align-items:center; justify-content:center;">
        <img src="results/2D reconstruction/sim reconstruction/eight_recon.gif" style="max-width:280px; max-height:280px; object-fit:contain;">
      </div>
    </td>
  </tr>
</table>

## 2D experimental model

The 2D experimental model was trained on a spiral trajectory. The trained model was then evaluated on different test trajectories to assess its generalisation capabilities. To test the robustness to velocity variations, an additional experiment was performed where the movement speed was increased by increasing the distance between each discrete point. A comparative analysis between model architectures with and without an LSTM layer was also performed to highlight the capability of the LSTM layer to model the time-dependent behavior of moving objects. The following figures show the results of the tests.

### Prediction of different trajectories

<table width="1000px" style="table-layout: fixed; background-color:#1a1a1a; color:white; border-collapse:collapse; margin-bottom:30px;">
  <tr>
    <th width="500px" style="text-align:center; padding:10px 0; border:1px solid #333; background-color:#1a1a1a; font-weight:bold;">
      Circle Trajectory
    </th>
    <th width="500px" style="text-align:center; padding:10px 0; border:1px solid #333; background-color:#1a1a1a; font-weight:bold;">
      Eight Trajectory
    </th>
  </tr>
  <tr>
    <td width="500px" height="300px" style="text-align:center; vertical-align:middle; padding:15px; border:1px solid #333;">
      <div style="height:280px; display:flex; align-items:center; justify-content:center;">
        <img src="results/2D reconstruction/exp reconstruction/lstm_circle_recon.gif" style="max-width:280px; max-height:280px; object-fit:contain;">
      </div>
    </td>
    <td width="500px" height="300px" style="text-align:center; vertical-align:middle; padding:15px; border:1px solid #333;">
      <div style="height:280px; display:flex; align-items:center; justify-content:center;">
        <img src="results/2D reconstruction/exp reconstruction/lstm_eight_recon.gif" style="max-width:280px; max-height:280px; object-fit:contain;">
      </div>
    </td>
  </tr>
  <tr>
    <th width="500px" style="text-align:center; padding:10px 0; border:1px solid #333; background-color:#1a1a1a; font-weight:bold;">
      Polynomial Trajectory
    </th>
    <th width="500px" style="text-align:center; padding:10px 0; border:1px solid #333; background-color:#1a1a1a; font-weight:bold;">
      Square Trajectory
    </th>
  </tr>
  <tr>
    <td width="500px" height="300px" style="text-align:center; vertical-align:middle; padding:15px; border:1px solid #333;">
      <div style="height:280px; display:flex; align-items:center; justify-content:center;">
        <img src="results/2D reconstruction/exp reconstruction/lstm_polynomial_recon.gif" style="max-width:280px; max-height:280px; object-fit:contain;">
      </div>
    </td>
    <td width="500px" height="300px" style="text-align:center; vertical-align:middle; padding:15px; border:1px solid #333;">
      <div style="height:280px; display:flex; align-items:center; justify-content:center;">
        <img src="results/2D reconstruction/exp reconstruction/lstm_square_recon.gif" style="max-width:280px; max-height:280px; object-fit:contain;">
      </div>
    </td>
  </tr>
</table>

### Prediction with different velocities

<table width="1000px" style="table-layout: fixed; background-color:#1a1a1a; color:white; border-collapse:collapse; margin-bottom:30px;">
  <tr>
    <th width="500px" style="text-align:center; padding:10px 0; border:1px solid #333; background-color:#1a1a1a; font-weight:bold;">
      Normal Velocity
    </th>
    <th width="500px" style="text-align:center; padding:10px 0; border:1px solid #333; background-color:#1a1a1a; font-weight:bold;">
      Increased Velocity
    </th>
  </tr>
  <tr>
    <td width="500px" height="300px" style="text-align:center; vertical-align:middle; padding:15px; border:1px solid #333;">
      <div style="height:280px; display:flex; align-items:center; justify-content:center;">
        <img src="results/2D reconstruction/exp reconstruction/lstm_eight_recon.gif" style="max-width:280px; max-height:280px; object-fit:contain;">
      </div>
    </td>
    <td width="500px" height="300px" style="text-align:center; vertical-align:middle; padding:15px; border:1px solid #333;">
      <div style="height:280px; display:flex; align-items:center; justify-content:center;">
        <img src="results/2D reconstruction/exp reconstruction/lstm_eight_fast_recon.gif" style="max-width:280px; max-height:280px; object-fit:contain;">
      </div>
    </td>
  </tr>
</table>

### Comparision of model with and without LSTM layer 

<table width="1000px" style="table-layout: fixed; background-color:#1a1a1a; color:white; border-collapse:collapse; margin-bottom:30px;">
  <tr>
    <th width="500px" style="text-align:center; padding:10px 0; border:1px solid #333; background-color:#1a1a1a; font-weight:bold;">
      With LSTM Layer
    </th>
    <th width="500px" style="text-align:center; padding:10px 0; border:1px solid #333; background-color:#1a1a1a; font-weight:bold;">
      Without LSTM Layer
    </th>
  </tr>
  <tr>
    <td width="500px" height="300px" style="text-align:center; vertical-align:middle; padding:15px; border:1px solid #333;">
      <div style="height:280px; display:flex; align-items:center; justify-content:center;">
        <img src="results/2D reconstruction/exp reconstruction/lstm_polynomial_recon.gif" style="max-width:280px; max-height:280px; object-fit:contain;">
      </div>
    </td>
    <td width="500px" height="300px" style="text-align:center; vertical-align:middle; padding:15px; border:1px solid #333;">
      <div style="height:280px; display:flex; align-items:center; justify-content:center;">
        <img src="results/2D reconstruction/exp reconstruction/no_lstm_polynomial_recon.gif" style="max-width:280px; max-height:280px; object-fit:contain;">
      </div>
    </td>
  </tr>
</table>

## 3D experimental model

The 3D experimental model was trained using a spiral helix trajectory with a radius that decreases with increasing height. Like the 2D experimental model, the 3D model was tested on various test trajectory (a normal helix trajectory and a circular sine wave). Different velocity variations were also tested and, finally, a comparison between the model with and without LSTM layer was performed. The following figures show the results of the tests.

### Prediction of different trajectories


<table width="1000px" style="table-layout: fixed; background-color:#1a1a1a; color:white; border-collapse:collapse; margin-bottom:30px;">
  <tr>
    <th width="500px" style="text-align:center; padding:10px 0; border:1px solid #333; background-color:#1a1a1a; font-weight:bold;">
     Helix Trajectory
    </th>
    <th width="500px" style="text-align:center; padding:10px 0; border:1px solid #333; background-color:#1a1a1a; font-weight:bold;">
      Circular Sine Wave Trajectory
    </th>
  </tr>
  <tr>
    <td width="500px" height="300px" style="text-align:center; vertical-align:middle; padding:15px; border:1px solid #333;">
      <div style="height:280px; display:flex; align-items:center; justify-content:center;">
        <img src="results/3D reconstruction/lstm_helix_recon.gif" style="max-width:280px; max-height:280px; object-fit:contain;">
      </div>
    </td>
    <td width="500px" height="300px" style="text-align:center; vertical-align:middle; padding:15px; border:1px solid #333;">
      <div style="height:280px; display:flex; align-items:center; justify-content:center;">
        <img src="results/3D reconstruction/lstm_circ_sine_recon.gif" style="max-width:280px; max-height:280px; object-fit:contain;">
      </div>
    </td>
  </tr>
</table>

### Prediction with different velocities

<table width="1000px" style="table-layout: fixed; background-color:#1a1a1a; color:white; border-collapse:collapse; margin-bottom:30px;">
  <tr>
    <th width="500px" style="text-align:center; padding:10px 0; border:1px solid #333; background-color:#1a1a1a; font-weight:bold;">
      Normal Velocity
    </th>
    <th width="500px" style="text-align:center; padding:10px 0; border:1px solid #333; background-color:#1a1a1a; font-weight:bold;">
      Increased Velocity
    </th>
  </tr>
  <tr>
    <td width="500px" height="300px" style="text-align:center; vertical-align:middle; padding:15px; border:1px solid #333;">
      <div style="height:280px; display:flex; align-items:center; justify-content:center;">
        <img src="results/3D reconstruction/lstm_helix_recon.gif" style="max-width:280px; max-height:280px; object-fit:contain;">
      </div>
    </td>
    <td width="500px" height="300px" style="text-align:center; vertical-align:middle; padding:15px; border:1px solid #333;">
      <div style="height:280px; display:flex; align-items:center; justify-content:center;">
        <img src="results/3D reconstruction/lstm_helix_fast_recon.gif" style="max-width:280px; max-height:280px; object-fit:contain;">
      </div>
    </td>
  </tr>
</table>

### Comparision of model with and without LSTM layer 

<table width="1000px" style="table-layout: fixed; background-color:#1a1a1a; color:white; border-collapse:collapse; margin-bottom:30px;">
  <tr>
    <th width="500px" style="text-align:center; padding:10px 0; border:1px solid #333; background-color:#1a1a1a; font-weight:bold;">
      With LSTM Layer
    </th>
    <th width="500px" style="text-align:center; padding:10px 0; border:1px solid #333; background-color:#1a1a1a; font-weight:bold;">
      Without LSTM Layer
    </th>
  </tr>
  <tr>
    <td width="500px" height="300px" style="text-align:center; vertical-align:middle; padding:15px; border:1px solid #333;">
      <div style="height:280px; display:flex; align-items:center; justify-content:center;">
        <img src="results/3D reconstruction/lstm_helix_recon.gif" style="max-width:280px; max-height:280px; object-fit:contain;">
      </div>
    </td>
    <td width="500px" height="300px" style="text-align:center; vertical-align:middle; padding:15px; border:1px solid #333;">
      <div style="height:280px; display:flex; align-items:center; justify-content:center;">
        <img src="results/3D reconstruction/no_lstm_helix_recon.gif" style="max-width:280px; max-height:280px; object-fit:contain;">
      </div>
    </td>
  </tr>
</table>
