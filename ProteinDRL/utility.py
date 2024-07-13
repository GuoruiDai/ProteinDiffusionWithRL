import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import subprocess
import glob
import local_dir
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation
from typing import NamedTuple
from graphtin import Graphtin
from PyQt5.QtWidgets import QApplication, QFileDialog


class Configuration(NamedTuple):
    '''Config for global settings; singleton ONLY'''
    model_save_path = local_dir.model_save_path
    training_data_folder = local_dir.training_data_folder

    min_aa_node: int
    max_aa_node: int # maximum number of amino acid nodes
    global_scaling: float # global scaling factor for all atom positions in R^3
    embed_dim: int # embedding dimension for node, edge, etc.
    diff_lr: float # diffusion learning rate
    rein_lr: float # reinforcement learning rate
    dropout_rate: float
    seed: int # seed for making PRNGKey
    device: torch.device
    discrete_num: int # number of discretizing steps
    save_interval: int # save after every k number of iterations
    rein_epoch_num: int # number of episodes per epoch for reinforcement learning


class ModelInputs(NamedTuple):
    '''Inputs of the model'''
    graph: Graphtin
    t: float | torch.Tensor


class TrajectoryStep(NamedTuple):
    '''Records values for a single step of a sampling trajectory'''
    t: float | torch.Tensor
    pos_t: torch.Tensor # t > s
    pos_s: torch.Tensor
    id_t: torch.Tensor
    id_s: torch.Tensor


class Utility():
    @staticmethod
    def torch_sinusoidal_embedding(input_scalar, embed_dim, device=torch.device('cuda')):
        half_dim = embed_dim // 2
        _intermediate = input_scalar / torch.pow(10000, (torch.arange(half_dim, device=device) / half_dim))
        output_array = torch.cat([torch.sin(_intermediate), torch.cos(_intermediate)], dim=-1)
        return output_array


    '''Creates shifted sinusoidal embedding for hidden nodes'''
    @staticmethod
    def create_initial_node_h_embedding(total_node, embed_dim, device=torch.device('cuda')):
        index_array = torch.reshape(torch.arange(total_node, device=device), (-1, 1))
        node_h_embedding = torch.vmap(Utility.torch_sinusoidal_embedding, in_dims=(0, None))(index_array, embed_dim)
        return node_h_embedding


    '''Creates shifted sinusoidal embedding for edges based on distance between each pair'''
    @staticmethod
    def create_initial_edge_embedding(distance_array, embed_dim):
        edge_embedding = torch.vmap(Utility.torch_sinusoidal_embedding, in_dims=(0, None))(distance_array*10, embed_dim)
        return edge_embedding


    @staticmethod
    def run_colabfold(input_sequence: str, seed = 123):
        #? empty the folder
        for item in os.listdir(local_dir.colab_result_dir):
            item_path = os.path.join(local_dir.colab_result_dir, item)
            try: os.remove(item_path)
            except: pass

        #? create the input file
        with open(local_dir.colab_result_dir + 'input_sequence.a3m', 'w') as file: 
            file.write('>\n' + input_sequence)

        #? run colabfold command
        command = (local_dir.colab_dir + 'colabfold_batch ' 
                + local_dir.colab_result_dir + 'input_sequence.a3m ' 
                + local_dir.colab_result_dir + ' '
                + '--model-type alphafold2 '
                + '--random-seed ' + str(seed) + ' '
                + '--num-recycle 2 '
                + '--amber --num-relax 1 --use-gpu-relax'
            )
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True) # suppress command log

        #? rename the ouput file
        old_file_path = Utility.find_file_path_with_partial_name(local_dir.colab_result_dir, 'input_sequence_relaxed')[0]
        os.rename(old_file_path, local_dir.colab_result_dir + 'relaxed_output.pdb')
        return local_dir.colab_result_dir + 'relaxed_output.pdb'


    @staticmethod
    def find_file_path_with_partial_name(dir, partial_name):
        return glob.glob(os.path.join(dir, f"*{partial_name}*"))


    @staticmethod
    def calculate_reward(score: float):
        '''Calculate reward value from the rosetta score'''
        return 1 - (2 / (1 + np.exp(- score * 0.1)))


    @staticmethod
    def _calculate_c_alpha_energy(distance: torch.Tensor | np.ndarray): #! old method
        optimal_distance = 4 #? optimal distance in angstrom
        d_zero = optimal_distance / (2**(1/2)) #? zero energy distance
        distance += 1e-8
        return 4 * ((d_zero / distance)**4 - (d_zero / distance)**2)

    
    @staticmethod
    def _calculate_energy_and_reward(input: torch.Tensor | np.ndarray, scaling_factor): #! old method
        if type(input) == np.ndarray:
            input = torch.from_numpy(input)
        aa_n_fr, aa_n_to = Graphtin.create_full_graph_node_index_arrays(input.shape[0])
        aa_pos_i, aa_pos_j = input[aa_n_to], input[aa_n_fr]
        aa_adj_distance = torch.sqrt(torch.sum((aa_pos_i - aa_pos_j)**2, dim=-1, keepdim=True))
        top_k_nn_distance = Graphtin.select_k_nearest_distance(aa_adj_distance, input.shape[0], k = 2)
        top_k_nn_energy = torch.vmap(Utility._calculate_c_alpha_energy, in_dims=(0, None))(top_k_nn_distance, scaling_factor)
        clamped_average_energy = torch.mean(torch.clamp_max(top_k_nn_energy, 2))
        reward = torch.expm1(-clamped_average_energy)
        return clamped_average_energy, reward
    

    @staticmethod
    def _check_model_grad(model):
        all_module_mean_grad_list = []
        for param in model.parameters():
            try:
                mean_grad = torch.mean(param.grad).item()
            except:
                pass
            all_module_mean_grad_list.append(mean_grad)
        print(all_module_mean_grad_list[-2])


    @staticmethod
    def _test_equivariance(model, config: Configuration):
        model.eval()
        rot_matrix = torch.from_numpy(Rotation.from_rotvec([1,2,4]).as_matrix()).to(device=config.device, dtype=torch.float32)
        reflec_matrix = torch.tensor([[-1., 0., 0.],
                                      [0., 1., 0.],
                                      [0., 0., 1.]]).to(device=config.device)
        
        phi = lambda matrix: (torch.matmul(matrix, rot_matrix)) # phi is a transformation on input x
        aa_n_fr, aa_n_to = Graphtin.create_full_graph_node_index_arrays(config.max_aa_node)
        _rand_x = torch.rand((config.max_aa_node, 3)).to(device=config.device)
        centralized_x = _rand_x - torch.mean(_rand_x, dim=0)
        graph = Graphtin(
            aa_n_fr = aa_n_fr,
            aa_n_to = aa_n_to,
            aa_pos_array = centralized_x,
            aa_mask = torch.ones((config.max_aa_node, 1)).to(device=config.device),
        )

        def path_1(): # x -> f(x) -> phi(f(x))
            f_x = model.diffusion_loss(ModelInputs(graph = graph, t = torch.ones((1,1), device=config.device)*0.2))
            return phi(f_x)

        def path_2(): # x -> phi(x) -> f(phi(x))
            phi_x = phi(graph.aa_pos_array)
            f_phi_x = model.diffusion_loss(ModelInputs(graph = graph._replace(aa_pos_array=phi_x), t = torch.ones((1,1), device=config.device)*0.2))
            return f_phi_x

        p_1 = path_1()
        p_2 = path_2()
        is_E3_equivariant = torch.allclose(p_1, p_2, atol= 1e-4)

        rot_matrix = torch.matmul(rot_matrix, reflec_matrix) #? invert x axis
        p_1 = path_1()
        p_2 = path_2()
        is_SE3_equivariant = not torch.allclose(p_1, p_2, atol= 1e-4)
        #? NOTE: SE(3) implies different result under 2 paths, since reflection breaks SE(3) symmetry

        print(f'E(3) Equivariance: {is_E3_equivariant}')
        print(f'SE(3) Equivariance: {is_SE3_equivariant}')
        exit()


    @staticmethod
    def visualize_backbone(df):
        if type(df) == pd.DataFrame:
            pass
        elif type(df) == torch.Tensor:
            df = np.asarray(df.detach().to('cpu'))
            df = pd.DataFrame(df)
        else:
            raise TypeError
        fig = px.line_3d(df, x=0, y=1, z=2)
        distances = np.sqrt(np.sum(np.diff(df, axis=0) ** 2, axis=1))
        cum_distances = np.insert(np.cumsum(distances), 0, 0)
        max_distance = np.max(cum_distances)
        fig.update_traces(line=dict(width=8, colorscale='Rainbow', color=cum_distances / max_distance))
        fig.add_scatter3d(
            x=df[0],
            y=df[1],
            z=df[2],
            mode="markers",
            marker=dict(size=4, color="gold"),
        )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="rgb(60, 60, 60)")
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
        fig.show()


    @staticmethod
    def select_file():
        app = QApplication(sys.argv)
        file_dialog = QFileDialog()
        file_dialog.setWindowTitle("Select File")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_paths = file_dialog.selectedFiles()
            return file_paths[0]  # Return the first selected file path
        else:
            return None
        

    @staticmethod
    def bit_thresholding(bit: torch.Tensor):
        bit[bit < 0] = -1
        bit[bit >= 0] = 1
        return bit

    
    @staticmethod
    class diffusion_step_recorder():
        def __init__(self, discrete_num):
            self.ndarray_step_list = [] # store the recorded coordinates as a list
            self.discrete_num = discrete_num

        def record_step(self, coord_t: torch.Tensor):
            coord_t = np.asarray(coord_t.detach().to('cpu'))
            self.ndarray_step_list.append(coord_t)
            
        def play_animation(self):
            fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
            fig.add_trace(go.Scatter3d())
            frames = []
            for i in range(self.discrete_num):
                coord = self.ndarray_step_list[i]
                distances = np.sqrt(np.sum(np.diff(coord, axis=0) ** 2, axis=1))
                cum_distances = np.insert(np.cumsum(distances), 0, 0)
                max_distance = np.max(cum_distances)
                frame = go.Frame(
                    data=[
                        go.Scatter3d(
                            x=coord[:, 0], y=coord[:, 1], z=coord[:, 2],
                            mode='lines',
                            line=dict(width=8, colorscale='Rainbow', color=cum_distances / max_distance),
                            name='Line'
                        ),
                        go.Scatter3d(
                            x=coord[:, 0], y=coord[:, 1], z=coord[:, 2],
                            mode="markers",
                            marker=dict(size=5, color="gold"),
                            name='Markers'
                        ),
                    ],
                    name=i
                )
                frames.append(frame)
            fig.frames = frames
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=0),
                paper_bgcolor="rgb(100, 100, 100)",
                scene=dict(
                    xaxis_visible=False,
                    yaxis_visible=False,
                    zaxis_visible=False
                ),
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "args": [None, {"frame": {"duration": 0, "redraw": True},
                                                "fromcurrent": False, "mode": "immediate",
                                                "transition": {"duration": 0}}],
                                "label": "Play",
                                "method": "animate",
                            },
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 87},
                        "showactive": False,
                        "type": "buttons",
                        "x": 0.1,
                        "xanchor": "right",
                        "y": 0,
                        "yanchor": "top",
                    }
                ],
                sliders=[{
                    "active": 0,
                    "steps": [{
                        "method": "animate",
                        "label": str(i),
                        "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    } for i in range(len(frames))]
                }],
            )
            fig.show()

