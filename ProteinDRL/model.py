import torch
import torch.nn as nn
import copy
from graphtin import Graphtin
from utility import Utility, Configuration, ModelInputs, TrajectoryStep


class VectorForwardLayer(nn.Module):
    def __init__(self, in_dim: int, config: Configuration, num_layer=6, attn=False, use_drp=True):
        super(VectorForwardLayer, self).__init__()
        self.num_layer = num_layer
        self.attn = attn
        self.use_drp = use_drp

        self.dropout_layer = nn.Dropout(p=config.dropout_rate)
        self.attention_layer = nn.MultiheadAttention(embed_dim=config.embed_dim, num_heads=8, dropout=config.dropout_rate)
        self.initial_embed = nn.Sequential(
            nn.Linear(in_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim),
        )
        self._forward_layer = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.LayerNorm(config.embed_dim),
        )
        self.forward_layer_stack = nn.ModuleList([copy.deepcopy(self._forward_layer) for _ in range(self.num_layer)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.use_drp == True:
            x = self.dropout_layer(x)
        x = self.initial_embed(x.masked_fill(mask, 0))
        if self.attn == True:
            x = x.masked_fill(mask, 0)
            x, _attn_weights = self.attention_layer(x, x, x)
        for layer in self.forward_layer_stack:
            x = x + 0.2 * layer(x.masked_fill(mask, 0))
        return x.masked_fill(mask, 0)


class SEGL(nn.Module):
    def __init__(self, config: Configuration):
        super(SEGL, self).__init__()
        self.config = config
        emb_dim = config.embed_dim

        self.embed_aa_dis = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.LayerNorm(16),
        )
        self.embed_time = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.LayerNorm(16),
        )
        self.embed_aa_scalar = VectorForwardLayer(3, config, num_layer=4, use_drp=False, attn=True)
        self.embed_aa_id = VectorForwardLayer(10, config, num_layer=4, use_drp=False, attn=True)
        self.aa_edge_forward_1 = VectorForwardLayer(emb_dim*5+32, config, num_layer=8, attn=True)
        self.aa_h_forward_1 = VectorForwardLayer(int(emb_dim*1+16), config, num_layer=4, attn=True)
        self.aa_edge_pos_forward = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 3),
        )
        self.aa_id_forward_1 = VectorForwardLayer(emb_dim+10, config, num_layer=6, attn=True)
        self.aa_id_final_forward = nn.Sequential(
            nn.Linear(emb_dim+5, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 5),
        )


    def forward(self, graph: Graphtin, t: torch.Tensor):
        num_edge = self.config.max_aa_node * (self.config.max_aa_node - 1)
        # These values need to be updated at the end of each layer
        aa_pos_array = graph.aa_pos_array
        aa_id_array = graph.aa_id_array
        aa_h_array = graph.aa_h_array
        aa_edge = graph.aa_edge

        aa_pos_i, _ = Graphtin.create_adjacency_array(aa_pos_array, graph.aa_n_to, graph.aa_mask)
        aa_pos_j, _ = Graphtin.create_adjacency_array(aa_pos_array, graph.aa_n_fr, graph.aa_mask)
        aa_id_i, _ = Graphtin.create_adjacency_array(aa_id_array, graph.aa_n_to, graph.aa_mask)
        aa_id_j, _ = Graphtin.create_adjacency_array(aa_id_array, graph.aa_n_fr, graph.aa_mask)
        aa_dis_ij = torch.sqrt((torch.sum((aa_pos_i - aa_pos_j)**2, dim=-1, keepdim=True)) + 1e-8)

        #? Create SE(3) frame and scalar tuple
        aa_adj_frame = Graphtin.frame(aa_pos_i, aa_pos_j)
        aa_adj_scalar_tuple = Graphtin.scalarization(aa_pos_i, aa_adj_frame)

        #? Embedding
        t_embed = self.embed_time(t)
        aa_adj_id_embed = self.embed_aa_id(torch.cat([aa_id_i, aa_id_j], dim=-1).masked_fill(graph.aa_adj_mask, 0), graph.aa_adj_mask)
        aa_adj_dis_embed = self.embed_aa_dis(aa_dis_ij.masked_fill(graph.aa_adj_mask, 0)).masked_fill(graph.aa_adj_mask, 0)
        aa_adj_scalar_embed = self.embed_aa_scalar(aa_adj_scalar_tuple, graph.aa_adj_mask)

        #? Graph Update
        aa_h_array_m = self.aa_h_forward_1(torch.cat([aa_h_array, t_embed.repeat(self.config.max_aa_node, 1)], dim=-1), graph.aa_mask)
        aa_h_array = aa_h_array + aa_h_array_m * 0.2

        aa_edge_m = self.aa_edge_forward_1(
            torch.cat([aa_edge, aa_adj_scalar_embed, aa_adj_id_embed, aa_adj_dis_embed, 
                       t_embed.repeat(num_edge, 1), aa_h_array[graph.aa_n_fr], aa_h_array[graph.aa_n_to]], dim=-1), graph.aa_adj_mask)
        aa_edge = aa_edge + aa_edge_m * 0.2

        #? Prediction
        aa_pos_adj_message = self.aa_edge_pos_forward(aa_edge).masked_fill(graph.aa_adj_mask, 0)
        aa_adj_vector = Graphtin.vectorization(aa_pos_adj_message, aa_adj_frame).masked_fill(graph.aa_adj_mask, 0)
        predicted_aa_pos_array = Graphtin.aggregate_neighbor_messages(aa_adj_vector, self.config.max_aa_node).masked_fill(graph.aa_mask, 0)

        aa_adj_id_array = self.aa_id_forward_1(torch.cat([aa_edge, aa_id_i, aa_id_j], dim=-1), graph.aa_adj_mask)
        aa_id_array_m = Graphtin.aggregate_neighbor_messages(aa_adj_id_array, self.config.max_aa_node).masked_fill(graph.aa_mask, 0)
        predicted_aa_id_array = self.aa_id_final_forward(torch.cat([aa_id_array_m, aa_id_array], dim=-1)).masked_fill(graph.aa_mask, 0)

        graph = graph._replace(aa_pos_array=predicted_aa_pos_array, aa_id_array=predicted_aa_id_array, aa_h_array=aa_h_array, aa_edge=aa_edge)
        return graph


class SEGN(nn.Module):
    '''SE(3) Graph Network'''
    def __init__(self, config: Configuration):
        super(SEGN, self).__init__()
        self.config = config
        self.SEGL_1 = SEGL(config)
        self.SEGL_2 = SEGL(config)

    def forward(self, graph: Graphtin, t):
        #? Initialize Graph Features
        aa_h_array = Utility.create_initial_node_h_embedding(self.config.max_aa_node, self.config.embed_dim).masked_fill(graph.aa_mask, 0)
        aa_pos_i, aa_mask_i = Graphtin.create_adjacency_array(graph.aa_pos_array, graph.aa_n_to, graph.aa_mask)
        aa_pos_j, aa_mask_j = Graphtin.create_adjacency_array(graph.aa_pos_array, graph.aa_n_fr, graph.aa_mask)
        aa_dis_ij = torch.sqrt((torch.sum((aa_pos_i - aa_pos_j)**2, dim=-1, keepdim=True)))
        aa_adj_mask = torch.logical_or(aa_mask_i, aa_mask_j)    
        initial_edge_embed = Utility.create_initial_edge_embedding(aa_dis_ij, self.config.embed_dim).masked_fill(aa_adj_mask, 0)
        graph = graph._replace(aa_h_array=aa_h_array, aa_edge=initial_edge_embed, aa_adj_mask=aa_adj_mask)

        #? Model Layers
        graph = self.SEGL_1(graph, t)
        graph = self.SEGL_2(graph, t)
        return (graph.aa_pos_array, graph.aa_id_array)



#?#######################################################################################################
class ProteinDiffusionModel(nn.Module):
    def __init__(self, config: Configuration):
        super(ProteinDiffusionModel, self).__init__()
        self.config = config
        self.SEGN = SEGN(config)

    def _gamma_scheduler(self, t):
        return 16 * t - 11


    def diffusion_loss(self, model_input: ModelInputs):
        t = model_input.t
        graph = model_input.graph
        # return self.SEGN(graph, t) #^: For Equivariance Test
        gamma_t = self._gamma_scheduler(t)
        var_t = torch.sigmoid(gamma_t)

        #? Random Sampling
        eps_pos = torch.normal(mean=0, std=1, size=(self.config.max_aa_node, 3), device=self.config.device).masked_fill(graph.aa_mask, 0)
        eps_id = torch.normal(mean=0, std=1, size=(self.config.max_aa_node, 5), device=self.config.device).masked_fill(graph.aa_mask, 0)

        #? Forward Diffusion
        aa_pos_t = torch.sqrt(1 - var_t) * graph.aa_pos_array + torch.sqrt(var_t) * eps_pos
        aa_id_t = torch.sqrt(1 - var_t) * graph.aa_id_array + torch.sqrt(var_t) * eps_id

        #? Model Predictions
        graph = graph._replace(aa_pos_array = aa_pos_t, aa_id_array = aa_id_t)
        eps_pos_hat, eps_id_hat = self.SEGN(graph, t)

        #? Loss Function
        position_loss_function = 0.5 * 16 * torch.sum(torch.square(eps_pos - eps_pos_hat)) / (3 * graph.seq_length)
        identity_loss_function = 0.5 * 16 * torch.sum(torch.square(eps_id - eps_id_hat)) / (5 * graph.seq_length)
        total_loss_function = 0.5 * (position_loss_function + identity_loss_function)
        return total_loss_function


    def reverse_step(self, model_input: ModelInputs):
        t = model_input.t
        graph = model_input.graph
        t_s = t - 1/self.config.discrete_num
        gamma_t, gamma_s = self._gamma_scheduler(t), self._gamma_scheduler(t_s)

        #? Random Sampling
        eps_pos = torch.normal(mean=0, std=1, size=(self.config.max_aa_node, 3), device=self.config.device).masked_fill(graph.aa_mask, 0)
        eps_id = torch.normal(mean=0, std=1, size=(self.config.max_aa_node, 5), device=self.config.device).masked_fill(graph.aa_mask, 0)

        #? Model Outputs
        eps_pos_hat, eps_id_hat = self.SEGN(graph, t)

        #? Reverse Step
        alpha_s_squared = torch.sigmoid(-gamma_s)
        alpha_t_squared = torch.sigmoid(-gamma_t)
        c = -torch.expm1(gamma_s - gamma_t)
        aa_pos_s = torch.sqrt(alpha_s_squared / alpha_t_squared) * (graph.aa_pos_array - c * torch.sqrt(torch.sigmoid(gamma_t)) * eps_pos_hat) \
                   + torch.sqrt((1 - alpha_s_squared) * c) * eps_pos
        aa_id_s = torch.sqrt(alpha_s_squared / alpha_t_squared) * (graph.aa_id_array - c * torch.sqrt(torch.sigmoid(gamma_t)) * eps_id_hat) \
                   + torch.sqrt((1 - alpha_s_squared) * c) * eps_id

        graph = graph._replace(aa_pos_array = aa_pos_s.masked_fill(graph.aa_mask, 0), aa_id_array = aa_id_s.masked_fill(graph.aa_mask, 0))
        return graph, eps_pos_hat, eps_id_hat


    def reinforce_loss_step(self, model_input: ModelInputs, trajectory_step: TrajectoryStep):
        '''The gradient of this function should be the negative policy gradient of the total reward,
        hence a gradient descent under this function is a gradient ascent under the total reward'''
        t = model_input.t
        graph = model_input.graph
        pos_t = trajectory_step.pos_t
        pos_s = trajectory_step.pos_s
        id_t = trajectory_step.id_t
        id_s = trajectory_step.id_s

        t_s = t - 1/self.config.discrete_num
        gamma_t, gamma_s = self._gamma_scheduler(t), self._gamma_scheduler(t_s)
        alpha_t_sqrd = torch.sigmoid(-gamma_t)
        alpha_t = torch.sqrt(alpha_t_sqrd)
        alpha_s = torch.sqrt(torch.sigmoid(-gamma_s))
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
        var_s = torch.sigmoid(gamma_s)
        c = -torch.expm1(gamma_s - gamma_t)

        #? surrogate loss function for the expected reward
        eps_pos_hat, eps_id_hat = self.SEGN(graph, t)
        position_loss_fn_t = 0.5 * (c*var_s) * (pos_s - (alpha_s/alpha_t) * (pos_t - sigma_t * c * eps_pos_hat))**2
        position_loss_fn_t = torch.sum(position_loss_fn_t) / (3 * graph.seq_length)
        identity_loss_fn_t = 0.5 * (c*var_s) * (id_s - (alpha_s/alpha_t) * (id_t - sigma_t * c * eps_id_hat))**2
        identity_loss_fn_t = torch.sum(identity_loss_fn_t) / (5 * graph.seq_length)
        total_loss_fn_t = 0.2 * position_loss_fn_t + 0.8 * identity_loss_fn_t
        return total_loss_fn_t