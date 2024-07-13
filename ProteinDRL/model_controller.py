import torch
import numpy as np
import os
import time
import pandas as pd
import aa_encoding
import local_dir
import pyrosetta
from torch.utils import data
from tqdm import tqdm
from utility import Utility, Configuration, ModelInputs, TrajectoryStep
from graphtin import Graphtin
from model import ProteinDiffusionModel


class ProteinDataPreparer(data.Dataset):
    def __init__(self, config: Configuration):
        self.data_folder = config.training_data_folder
        self.data_list = tuple(os.walk(config.training_data_folder))[0][2]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        file_name: str = self.data_list[index]
        file_path = self.data_folder + '/' + file_name
        raw_data = np.load(file_path)['data']
        return raw_data

class GraphtinLoader(data.DataLoader):
    def __init__(self, dataset, config: Configuration):
        self.config = config
        self.aa_bit_converter = aa_encoding.aa_bit_converter()
        super(self.__class__, self).__init__(
            dataset, collate_fn=self.process_raw_data, batch_size=None, shuffle=True, pin_memory=True, prefetch_factor=4, num_workers=8)
    
    def process_raw_data(self, raw_data: np.ndarray):
        seq_length = raw_data.shape[0]
        raw_position = raw_data[:, 0:3].astype(float)
        raw_identity = raw_data[:, 3]
        aa_mask = torch.cat([torch.zeros(seq_length, 1), torch.ones(self.config.max_aa_node - seq_length, 1)], dim=0).to(bool) #? 1 = mask; 0 = no mask        

        padded_position = np.concatenate([raw_position, np.zeros(((self.config.max_aa_node - seq_length), raw_position.shape[1]))], axis=0)
        aa_position = torch.from_numpy(padded_position).to(torch.float32) * self.config.global_scaling
        #? Centralize protein; don't use simple average, since different proteins have different length
        aa_position = (aa_position - torch.sum(aa_position, dim=0, keepdim=True)/seq_length).masked_fill(aa_mask, 0)

        encoded_identity = np.apply_along_axis(self.aa_bit_converter.convert_aa_to_bit, 1, raw_identity.reshape(-1, 1))
        aa_identity_np = np.concatenate([encoded_identity, np.zeros(((self.config.max_aa_node - seq_length), encoded_identity.shape[1]))], axis=0)
        aa_identity = torch.from_numpy(aa_identity_np).to(torch.float32)
        return aa_position, aa_identity, aa_mask, seq_length


class ModelController():
    '''Main class for handling model training etc.; do not create instance'''
    config = Configuration(
        min_aa_node = 20,
        max_aa_node = 80,
        global_scaling = 0.1,
        embed_dim = 320,
        diff_lr = 2e-4,
        rein_lr = 5e-6,
        dropout_rate = 0.1,
        seed = 0,
        device = torch.device('cuda'),
        discrete_num = 100,
        save_interval = 10000,
        rein_epoch_num = 100,
    )
    protein_data_fetcher = ProteinDataPreparer(config)
    graphtin_loader = GraphtinLoader(protein_data_fetcher, config)
    aa_bit_converter = aa_encoding.aa_bit_converter()
    pyrosetta.init()
    torch.manual_seed(config.seed)
    torch.cuda.empty_cache()
    torch.set_printoptions(precision=4, sci_mode=False) # set number of decimals when printing tensors

    @classmethod
    def save_model(self, model, step, diff_optimizer=None, rl_optimizer=None):
        diff_optim_dict, rl_optim_dict = None, None
        if diff_optimizer != None:
            diff_optim_dict = diff_optimizer.state_dict()
        if rl_optimizer != None:
            rl_optim_dict = rl_optimizer.state_dict()
        torch.save({
            'model_state_dict': model.state_dict(),
            'diff_optim_state_dict': diff_optim_dict,
            'rl_optim_state_dict': rl_optim_dict,
            'step': step,
            }, self.config.model_save_path + '/model.pt')

    @classmethod
    def load_model(self, model):
        model_save = torch.load(self.config.model_save_path + '/model.pt')
        model.load_state_dict(model_save['model_state_dict'])
        step = int(model_save['step'])
        print('Model Loaded')
        return model, step
    

    @classmethod
    def fold_sequence_and_calculate_rosetta_score(self, input_sequence: str, seed):
        output_pdb_path = Utility.run_colabfold(input_sequence, seed)
        pose = pyrosetta.pose_from_pdb(output_pdb_path)
        rosetta_score = pyrosetta.get_fa_scorefxn()(pose)
        return rosetta_score
    

    @classmethod
    def torch_id_bit_to_sequence(self, torch_bit_array): #? convert amino acid bit array to sequence as string on cpu
        sample_id_numpy = Utility.bit_thresholding(torch_bit_array).detach().cpu().numpy()
        aa_id_array = np.apply_along_axis(self.aa_bit_converter.convert_aa_to_bit, 1, sample_id_numpy, True)
        aa_sequence = ''.join(aa_id_array)
        return aa_sequence


    @classmethod
    def train(self):
        model = ProteinDiffusionModel(self.config).to(device=self.config.device)
        diff_optimizer = torch.optim.Adam(model.parameters(), lr=self.config.diff_lr)
        try:
            model, step = self.load_model(model)
            diff_optimizer.load_state_dict(torch.load(self.config.model_save_path + '/model.pt')['diff_optim_state_dict'])
        except:
            step = 0
            print('New Model Initialized')

        # Utility._test_equivariance(model, self.config) #^ For Testing Equivariance
        model.train()
        loss_record = 0
        progress_bar = tqdm(total=self.config.save_interval, leave=True, colour='blue')

        # torch.autograd.set_detect_anomaly(True)
        while True:
            for data in self.graphtin_loader:
                aa_coordinate, aa_identity, aa_mask, seq_length = data
                aa_n_fr, aa_n_to = Graphtin.create_full_graph_node_index_arrays(self.config.max_aa_node)
                graph = Graphtin(
                    seq_length = seq_length,
                    aa_n_fr = aa_n_fr,
                    aa_n_to = aa_n_to,
                    aa_pos_array = aa_coordinate.to(device=self.config.device),
                    aa_id_array = aa_identity.to(device=self.config.device),
                    aa_mask = aa_mask.to(device=self.config.device),
                )

                t = torch.rand((1, 1)).to(device=self.config.device)
                model_input = ModelInputs(graph = graph, t = t)
                
                diff_optimizer.zero_grad()
                loss_fn = model.diffusion_loss(model_input)
                loss_fn.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2) # gradient clipping

                # Utility._check_model_grad(model) #^
                diff_optimizer.step()

                loss_record += loss_fn.detach().item()
                step += 1
                progress_bar.update(n=1)

                if step % self.config.save_interval == 0:
                    self.save_model(model, step, diff_optimizer=diff_optimizer)
                    progress_bar.reset(total=self.config.save_interval)
                    average_loss = round(loss_record / self.config.save_interval, 4)
                    loss_record = 0

                    new_row = ['Step: ' + str(step)] + ['Loss: ' + str(average_loss)]
                    new_row_df = pd.DataFrame([new_row])
                    new_row_df.to_csv(self.config.model_save_path+'/model_log.txt', index=False, mode='a', header=False, sep='\t')
                    time.sleep(self.config.save_interval * 0.02)


    @classmethod
    def reinforce(self):
        @torch.no_grad()
        def sample_trajectory(model: ProteinDiffusionModel, graph: Graphtin):
            '''Sample one complete trajectory'''
            complete_trajectory: list[TrajectoryStep] = []
            model.eval()
            for t in range(self.config.discrete_num, 0, -1):
                t = torch.tensor([[t / self.config.discrete_num]]).to(device=self.config.device)
                pos_t = graph.aa_pos_array # save a copy before taking a reverse step
                id_t = graph.aa_id_array
                model_input = ModelInputs(
                    graph = graph,
                    t = t,
                )
                graph, _, _ = model.reverse_step(model_input)
                trajectory_step_i = TrajectoryStep(
                    t = t,
                    pos_t = pos_t,
                    pos_s = graph.aa_pos_array,
                    id_t = id_t,
                    id_s = graph.aa_id_array,
                )
                complete_trajectory.append(trajectory_step_i)
            return complete_trajectory

        def update_policy(model: ProteinDiffusionModel, graph: Graphtin, complete_trajectory: list[TrajectoryStep], 
                          optimizer: torch.optim.Adam, baseline: torch.Tensor):
            '''Accumulate gradients of all steps then update'''
            model.train()
            optimizer.zero_grad()

            pos_hat_0 = complete_trajectory[-1].pos_s * (1/self.config.global_scaling)
            id_bit_hat_0 = complete_trajectory[-1].id_s[0:graph.seq_length, :]
            for trajectory_step_i in complete_trajectory:
                t = trajectory_step_i.t
                pos_t = trajectory_step_i.pos_t
                id_t = trajectory_step_i.id_t
                model_input = ModelInputs(
                    graph = graph._replace(aa_pos_array = pos_t, aa_id_array = id_t),
                    t = t,
                )
                loss_fn = model.reinforce_loss_step(model_input, trajectory_step_i)
                loss_fn.backward() # accumulate gradient for each step

            # _, reward = Utility._calculate_energy_and_reward(pos_hat_0[0:graph.seq_length, :], self.config.global_scaling)
            sample_sequence = self.torch_id_bit_to_sequence(id_bit_hat_0)
            rosetta_score = self.fold_sequence_and_calculate_rosetta_score(sample_sequence, self.config.seed)
            reward = Utility.calculate_reward(rosetta_score)
            reward = torch.tensor(reward, device=self.config.device).to(torch.float32)

            print(sample_sequence)
            print(rosetta_score)

            advantage = reward - baseline
            loss_fn *= advantage

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.6, norm_type=2)
            optimizer.step() # update params after an entire trajectory
            return reward.detach().item(), advantage.detach().item()

        print('Begin Reinforcement')
        model: ProteinDiffusionModel = ProteinDiffusionModel(self.config).to(device=self.config.device)
        model, step = self.load_model(model)
        reinforce_optimizer = torch.optim.Adam(model.parameters(), lr=self.config.rein_lr)
        try:
            reinforce_optimizer.load_state_dict(torch.load(self.config.model_save_path + '/model.pt')['rl_optim_state_dict'])
        except:
            pass
        reinforce_progress_bar = tqdm(total=self.config.rein_epoch_num, leave=True, colour='blue')
        reward_record, advantage_record = 0, 0

        max_episode = 2000
        total_sum_baseline = torch.tensor(-0.5, dtype=torch.float32, device=self.config.device) # set initial baseline value
        for i in range(max_episode):
            seq_length = torch.randint(low=self.config.min_aa_node, high=self.config.max_aa_node+1, size=(1,))
            aa_mask = torch.where(torch.arange(self.config.max_aa_node) < seq_length, 0, 1).view(-1, 1).to(bool)
            aa_position_T_max_sample = torch.normal(0, 1, size=(self.config.max_aa_node, 3)).masked_fill(aa_mask, 0)
            aa_identity_T_max_sample = torch.normal(0, 1, size=(self.config.max_aa_node, 5)).masked_fill(aa_mask, 0)
            aa_n_fr, aa_n_to = Graphtin.create_full_graph_node_index_arrays(self.config.max_aa_node)
            graph: Graphtin = Graphtin(
                seq_length = seq_length.to(device=self.config.device),
                aa_n_fr = aa_n_fr,
                aa_n_to = aa_n_to,
                aa_pos_array = aa_position_T_max_sample.to(device=self.config.device),
                aa_id_array = aa_identity_T_max_sample.to(device=self.config.device),
                aa_mask = aa_mask.to(device=self.config.device),
            )
            complete_trajectory = sample_trajectory(model, graph)
            mean_baseline = total_sum_baseline/(i + 1)
            reward, advantage = update_policy(model, graph, complete_trajectory, reinforce_optimizer, mean_baseline)
            total_sum_baseline += reward
            reward_record += reward
            advantage_record += advantage
            step += 1
            reinforce_progress_bar.update(n=1)

            if step % self.config.rein_epoch_num == 0:
                self.save_model(model, step, rl_optimizer=reinforce_optimizer)
                reinforce_progress_bar.reset(total = self.config.rein_epoch_num)
                average_reward = round(reward_record / self.config.rein_epoch_num, 4)
                average_advantage = round(advantage_record / self.config.rein_epoch_num, 4)
                reward_record, advantage_record = 0, 0
                new_row = ['Reward: ' + str(average_reward)] + ['Advantage: ' + str(average_advantage)]
                new_row_df = pd.DataFrame([new_row])
                new_row_df.to_csv(self.config.model_save_path+'/model_reinforce.txt', index=False, mode='a', header=False, sep='\t')
                time.sleep(self.config.rein_epoch_num * 0.5)


    @classmethod
    @torch.no_grad()
    def sample(self):
        model: ProteinDiffusionModel = ProteinDiffusionModel(self.config).to(device=self.config.device)
        model, _ = self.load_model(model)
        model.eval()
        diffusion_recorder = Utility.diffusion_step_recorder(discrete_num=self.config.discrete_num)
        progress_bar = tqdm(total=self.config.discrete_num, leave=True, colour='blue')

        def _sample_T_max(seq_length):
            aa_mask = torch.cat([torch.zeros(seq_length, 1), torch.ones(self.config.max_aa_node - seq_length, 1)], dim=0).to(bool)
            aa_position_T_max_sample = torch.normal(0, 1, size=(self.config.max_aa_node, 3)).masked_fill(aa_mask, 0)
            aa_identity_T_max_sample = torch.normal(0, 1, size=(self.config.max_aa_node, 5)).masked_fill(aa_mask, 0)
            aa_n_fr, aa_n_to = Graphtin.create_full_graph_node_index_arrays(self.config.max_aa_node)
            graph = Graphtin(
                seq_length = seq_length,
                aa_n_fr = aa_n_fr,
                aa_n_to = aa_n_to,
                aa_pos_array = aa_position_T_max_sample.to(device=self.config.device),
                aa_id_array = aa_identity_T_max_sample.to(device=self.config.device),
                aa_mask = aa_mask.to(device=self.config.device),
            )
            return graph

        def _reverse_step_loop(graph_t_max) -> Graphtin:
            progress_bar.reset(total=self.config.discrete_num)
            graph = graph_t_max
            for t in range(self.config.discrete_num, 0, -1): # from T_max to 1
                t = torch.tensor([[t / self.config.discrete_num]]).to(device=self.config.device) # scale range to [0,1]
                model_input = ModelInputs(
                    graph = graph,
                    t = t,
                )
                graph, _, _ = model.reverse_step(model_input)
                diffusion_recorder.record_step(coord_t = graph.aa_pos_array)
                progress_bar.update(n=1)
            return graph

        def sample_one_protein(seq_length):
            graph_at_t_max = _sample_T_max(seq_length)
            sampled_graph = _reverse_step_loop(graph_at_t_max)
            sampled_position = sampled_graph.aa_pos_array[0:seq_length, :] * (1/self.config.global_scaling)
            sampled_identity = sampled_graph.aa_id_array[0:seq_length, :]
            return sampled_position, sampled_identity
        
        def save_sample_to_disk(sample):
            sample = np.asarray(sample.detach().item())
            sample_df = pd.DataFrame(sample)
            sample_df.to_csv(self.config.model_save_path+'/generated_protein.txt', sep='\t', index=False, header=False)

        def record_energy_values(energy):
            new_row_df = pd.DataFrame([energy])
            new_row_df.to_csv(self.config.model_save_path+'/energy_values.txt', index=False, mode='a', header=False, sep='\t')


        for i in range(6):
            seed = i + 100 + 26
            torch.random.manual_seed(seed)
            random_seq_length = torch.randint(low=self.config.min_aa_node, high=self.config.max_aa_node+1, size=(1,))
            sample_position, sample_identity = sample_one_protein(random_seq_length)
            # energy, _ = Utility.calculate_energy_and_reward(sample_position[0:random_seq_length, :], self.config.global_scaling)

            sample_sequence = self.torch_id_bit_to_sequence(sample_identity)
            
            #? fold sequence and calculate rosetta score
            rosetta_score = self.fold_sequence_and_calculate_rosetta_score(sample_sequence, seed)

            record_energy_values(round(rosetta_score, 6))
            Utility.visualize_backbone(sample_position)
            diffusion_recorder.play_animation()
            save_sample_to_disk(sample_sequence)



ModelController.train()
# ModelController.reinforce()
# ModelController.sample()

