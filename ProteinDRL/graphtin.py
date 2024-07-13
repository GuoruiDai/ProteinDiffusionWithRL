import torch
import torch.nn.functional as F
from typing import NamedTuple

class Graphtin(NamedTuple):
    '''
    Graphtin: Graph class for protein data. This class provides the abstraction and some graph related operations. \n
    '''

    seq_length: torch.Tensor = None # this length is the input protein's length
    aa_n_fr: torch.Tensor = None
    aa_n_to: torch.Tensor = None
    aa_pos_array: torch.Tensor = None
    aa_id_array: torch.Tensor = None
    aa_h_array: torch.Tensor = None    
    aa_mask: torch.Tensor = None
    aa_edge: torch.Tensor = None
    aa_adj_mask: torch.Tensor = None


    @staticmethod
    def create_full_graph_node_index_arrays(total_node: int, device=torch.device('cuda')):
        '''
        Create fully connected source & destination array without self-loop
        example of a 3-node graph:
            source      = [1 2 0 2 0 1]
            destination = [0 0 1 1 2 2]
        '''
        full_node_index_vector = torch.arange(0, total_node, device=device)
        full_node_broadcast_array = full_node_index_vector.repeat(total_node, 1)
        inverse_I_mask = (1-torch.eye(total_node, device=device)).bool()
        source_2D_array = full_node_broadcast_array[inverse_I_mask]
        destination = full_node_index_vector.repeat_interleave(total_node - 1)
        return source_2D_array, destination
    

    @staticmethod
    def update_graph(input_array: torch.Tensor, total_node: torch.Tensor, n_fr: torch.Tensor, n_to: torch.Tensor, 
                     mask: torch.Tensor, node_array: torch.Tensor = None, adj_array: torch.Tensor = None):
        '''Base function for updating graph nodes or edges. All feature arrays must be 2D, and it will be updated by the 
           node array or edge array if given. This function will automatically infer whether the input is a node array or edge/adjacency array, 
           then update it accordingly. \n
           NOTE: This function assumes a fully connected and undirected graph without self-loop.'''
        assert input_array.dim() == 2
        num_edge = total_node * (total_node - 1)
        adj_array_mask = mask[n_fr] * mask[n_to]

        if input_array.shape[0] == total_node: # if input is a node array
            if node_array is not None:
                node_message_array = input_array[n_fr]
            elif node_array is None:
                node_message_array = 0
            
            if adj_array is not None:
                adj_message_array = adj_array
            elif adj_array is None:
                adj_message_array = 0

            new_message_array = (node_message_array + adj_message_array) * adj_array_mask
            new_message_array = Graphtin.aggregate_neighbor_messages(new_message_array, total_node)
            input_array = 0.7 * (input_array + new_message_array * 0.5)
        
        elif input_array.shape[0] == num_edge: # if input is an edge or node-adjacent array
            if node_array is not None:
                self_adj_array = input_array[n_to]
                neighbor_adj_array = input_array[n_fr]
                node_message_array = neighbor_adj_array + self_adj_array
            elif node_array is None:
                node_message_array = 0

            if adj_array is not None:
                adj_message_array = adj_array
            elif adj_array is None:
                adj_message_array = 0

            new_message_array = (node_message_array + adj_message_array) * adj_array_mask
            input_array = 0.7 * (input_array + new_message_array * 0.5)
        else:
            raise ValueError
        return input_array
    

    @staticmethod
    def create_adjacency_array(input_array: torch.Tensor, index_array: torch.Tensor, mask: torch.Tensor = None):
        '''
        NOTE: This function assumes that the destination node indices (not n_to) are ordered from 0 to N-1 \n
        Given an array A and 1D index array, create an adjacency array; this is NOT a paired adjacency array, but rather only
            either the source or destination array. \n
        For example: input_array = [[1,2], [5,6]], index_array = [0,0,1,1], output_array = [[1,2], [1,2], [5,6], [5,6]] \n
        If a mask is provided, also create an adjacency array for the mask using the same index_array.
        '''
        mask_adjacency_array = None
        if mask is not None:
            assert mask.dim() == 2
            mask_adjacency_array = mask[index_array]

        assert input_array.dim() == 2
        adjacency_array = input_array[index_array]
        return adjacency_array, mask_adjacency_array


    @staticmethod
    def aggregate_neighbor_messages(input_adj_array: torch.Tensor, total_node: int):
        segment_length = input_adj_array.size(0) // total_node
        input_adj_tensor_3 = torch.reshape(input_adj_array, (total_node, segment_length, input_adj_array.shape[-1]))
        return torch.mean(input_adj_tensor_3, dim = 1)


    @staticmethod
    def frame(p_i, p_j) -> tuple:
        _p_j_minus_p_i = p_j - p_i
        f_a = _p_j_minus_p_i / (torch.norm(_p_j_minus_p_i, dim=-1, keepdim=True) + 1e-8)
        _p_j_cross_p_i = torch.cross(p_j, p_i, dim=-1)
        f_b = _p_j_cross_p_i / (torch.norm(_p_j_cross_p_i, dim=-1, keepdim=True) + 1e-8)
        f_c = torch.cross(f_a, f_b, dim=-1)
        frame = (f_a, f_b, f_c)
        return frame
    
    @staticmethod
    def scalarization(p_i, frame: tuple):
        f_a, f_b, f_c = frame
        s_a = torch.sum(f_a * p_i, dim=-1, keepdim=True)
        s_b = torch.sum(f_b * p_i, dim=-1, keepdim=True)
        s_c = torch.sum(f_c * p_i, dim=-1, keepdim=True)
        scalar = torch.cat([s_a, s_b, s_c], dim=-1)
        return scalar
    
    @staticmethod
    def vectorization(v: torch.Tensor, frame: tuple):
        v_a = v[:, 0].reshape(-1, 1)
        v_b = v[:, 1].reshape(-1, 1)
        v_c = v[:, 2].reshape(-1, 1)
        f_a, f_b, f_c = frame
        out_vector = v_a * f_a + v_b * f_b + v_c * f_c
        return out_vector
    

    @staticmethod
    def select_k_nearest_distance(pairwise_distance_array: torch.Tensor, total_node: int, k: int) -> torch.Tensor:
        dis_2D_array = pairwise_distance_array.view(total_node, total_node - 1)
        def pick_top_k_element(dis_array, k):
            topk_array, _ = torch.topk(-dis_array, k)
            return -topk_array
        topk_2D_array = torch.vmap(pick_top_k_element, in_dims=(0, None))(dis_2D_array, k)
        return topk_2D_array.reshape(-1, 1)


