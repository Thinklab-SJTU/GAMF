import torch, \
    ground_metric_gm as gm, \
    gurobi_qap_prior as gb, \
    model_gm as model
import numpy as np
from torchsummary import summary
import scipy.sparse as ss


def total_node_num(network: torch.nn.Module):
    '''
    count the total number of nodes in the network [network]
    '''
    num_nodes = 0
    for idx, (name, parameters) in enumerate(network.named_parameters()):
        if 'bias' in name:
            continue
        if idx == 0:
            num_nodes += parameters.shape[1]
        num_nodes += parameters.shape[0]
    return num_nodes


def graph_matching_fusion_slice(args, networks: list, sparse=False):
    '''
    the function is based on graph_matching_fusion() written by chenfei, and the core idea is to build the association
    graph with several adjacent layers (e.g. 3) instead of using the whole neural network for matching.
    We believe this approach will significantly reduce the time consuming for solving the QAP problem.
    '''
    # if args.device == torch.device('cpu'):
    #     networks[0] = networks[0].cpu()
    #     networks[1] = networks[1].cpu()
    n1 = total_node_num(network=networks[0])
    n2 = total_node_num(network=networks[1])
    assert (n1 == n2)
    affinity = None
    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    num_nodes_before = 0
    num_nodes_incremental = []
    num_nodes_layers = []
    pre_conv_list = []
    cur_conv_list = []
    conv_kernel_size_list = []
    num_nodes_pre = 0
    num_nodes_cur = 0
    is_conv = False
    pre_conv = False
    pre_conv_kernel_size = None
    pre_conv_out_channel = 1
    is_bias = False
    is_final_bias = False
    perm_is_complete = True
    named_weight_list_0 = [named_parameter for named_parameter in networks[0].named_parameters()]
    solution_forward_list = []
    solution_backward_list = []
    affinity_list = []
    for idx, ((_, fc_layer0_weight), (_, fc_layer1_weight)) in \
            enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):
        assert fc_layer0_weight.shape == fc_layer1_weight.shape
        layer_shape = fc_layer0_weight.shape
        num_nodes_cur = fc_layer0_weight.shape[0]
        if len(layer_shape) > 1:
            # if it's a fully-connected layer after a convolutional layer
            if is_conv is True and len(layer_shape) == 2:
                num_nodes_pre = pre_conv_out_channel
            else:
                num_nodes_pre = fc_layer0_weight.shape[1]
        # tell whether the layer is convolutional or fully-connected or bias
        if idx >= 1 and len(named_weight_list_0[idx - 1][1].shape) == 1:
            pre_bias = True
        else:
            pre_bias = False
        if len(layer_shape) > 2:
            is_bias = False
            if not pre_bias:
                pre_conv = is_conv
                pre_conv_list.append(pre_conv)
            is_conv = True
            cur_conv_list.append(is_conv)
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            fc_layer0_weight_data = fc_layer0_weight.data.view(
                fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(
                fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
        elif len(layer_shape) == 2:
            is_bias = False
            if not pre_bias:
                pre_conv = is_conv
                pre_conv_list.append(pre_conv)
            is_conv = False
            cur_conv_list.append(is_conv)
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data
        else:
            is_bias = True
            if not pre_bias:
                pre_conv = is_conv
                pre_conv_list.append(pre_conv)
            is_conv = False
            cur_conv_list.append(is_conv)
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data
        # if it's conv, update [pre_conv_out_channel]
        if is_conv:
            pre_conv_out_channel = num_nodes_cur
        # tell whether it's the final bias layer
        if is_bias is True and idx == num_layers - 1:
            is_final_bias = True
        # if it's the first layer, map the input nodes
        n1 = num_nodes_pre + num_nodes_cur
        n2 = num_nodes_pre + num_nodes_cur

        # affinity = torch.zeros([n1 * n2, n1 * n2], device=args.device)
        if not sparse:
            affinity = np.zeros([n1 * n2, n1 * n2])
        else:
            affinity = ss.lil_matrix((n1 * n2, n1 * n2))

        if idx == 0:
            s = torch.zeros([num_nodes_pre, num_nodes_pre], device=args.device)
            for a in range(num_nodes_pre):
                affinity[a * n2 + a, a * n2 + a] = 1
                s[a][a] = 1
            solution_forward_list.append(s)
        # if idx > 0:
        #     s = solution_list[-1].cpu().numpy()
        #     assert len(s) == num_nodes_pre
        # for i in range(len(s)):
        #     for j in range(len(s)):
        #         if s[i][j] == 1:
        #             affinity[i * n2 + j][i * n2 + j] = 1
        # if it's the final layer, map the output nodes
        if idx == num_layers - 2 and 'bias' in named_weight_list_0[idx + 1][0] or \
                idx == num_layers - 1 and 'bias' not in named_weight_list_0[idx][0]:
            s = torch.zeros([num_nodes_cur, num_nodes_cur], device=args.device)
            for a in range(num_nodes_cur):
                affinity[(num_nodes_pre + a) * n2 + num_nodes_pre + a, (num_nodes_pre + a) * n2 + num_nodes_pre + a] = 1
                s[a][a] = 1
            solution_backward_list.append(s)
        # calculate the edge-wise soft affinities between two models
        if is_bias is False:
            ground_metric = gm.Ground_Metric_GM(
                fc_layer0_weight_data, fc_layer1_weight_data, is_conv, is_bias,
                pre_conv, int(fc_layer0_weight_data.shape[1] / pre_conv_out_channel))
        else:
            ground_metric = gm.Ground_Metric_GM(
                fc_layer0_weight_data, fc_layer1_weight_data, is_conv, is_bias,
                pre_conv, 1)
        layer_affinity = ground_metric.process_soft_affinity(p=2)
        if is_bias is False:
            pre_conv_kernel_size = fc_layer0_weight.shape[3] if is_conv else None
            conv_kernel_size_list.append(pre_conv_kernel_size)
        # copy the affinity values from [layer_affinity] to the corresponding positions in [affinity] matrix
        if is_bias is True and is_final_bias is False:
            for a in range(num_nodes_cur):
                for c in range(num_nodes_cur):
                    affinity[a * n2 + c, a * n2 + c] = layer_affinity[a][c].cpu().detach().numpy()
        elif is_final_bias is False:
            for a in range(num_nodes_pre):
                for b in range(num_nodes_cur):
                    affinity[a * n2: a * n2 + num_nodes_pre,
                    (num_nodes_pre + b) * n2 + num_nodes_pre: (num_nodes_pre + b) * n2 + num_nodes_pre + num_nodes_cur] \
                        = layer_affinity[a + b * num_nodes_pre].view(num_nodes_cur, num_nodes_pre).transpose(0,
                                                                                                             1).cpu().detach().numpy()
                    # affinity[
                    # (num_nodes_pre + b) * n2 + num_nodes_pre: (num_nodes_pre + b) * n2 + num_nodes_pre + num_nodes_cur,
                    # a * n2: a * n2 + num_nodes_pre] = layer_affinity[a + b * num_nodes_pre].view(num_nodes_cur,
                    #                                                                              num_nodes_pre).cpu().detach().numpy()
        # update the total number of nodes that has already been considered in previous steps
        if is_bias is False:
            num_nodes_before += num_nodes_pre
            num_nodes_incremental.append(num_nodes_before)
            num_nodes_layers.append(num_nodes_cur)

        if idx < num_layers / 2:
            solution = gb.gurobi_qap_solver_prior(affinity, n1, n2, solution_forward_list[-1].cpu().detach().numpy(),
                                                  prior_position=0, device=args.device,
                                                  time_limit=30)
            s_cur = solution[num_nodes_pre: n1, num_nodes_pre: n2]
            s_pre = solution[0: num_nodes_pre, 0: num_nodes_pre]
            assert (s_pre.cpu().detach().numpy() == solution_forward_list[-1].cpu().detach().numpy()).all()
            solution_forward_list.append(s_cur)
        else:
            affinity_list.append(affinity)

    for i in range(len(affinity_list)):
        affinity = affinity_list[len(affinity_list) - i - 1]
        num_nodes_cur = num_nodes_layers[len(num_nodes_layers) - i - 1]
        num_nodes_pre = 1 if len(num_nodes_layers) - i - 1 == 0 else num_nodes_layers[len(num_nodes_layers) - i - 2]
        n1 = n2 = num_nodes_pre + num_nodes_cur
        solution = gb.gurobi_qap_solver_prior(affinity, n1, n2, solution_backward_list[-1].cpu().detach().numpy(),
                                              prior_position=1, device=args.device,
                                              time_limit=30)
        s_pre = solution[num_nodes_pre: n1, num_nodes_pre: n2]
        s_cur = solution[0: num_nodes_pre, 0: num_nodes_pre]
        assert (s_pre.cpu().detach().numpy() == solution_backward_list[-1].cpu().detach().numpy()).all()
        solution_backward_list.append(s_cur)

    aligned_wt_0 = [parameter.data for name, parameter in named_weight_list_0]
    idx = 0
    num_layers = len(aligned_wt_0)
    for num_before, num_cur, cur_conv, cur_kernel_size in \
            zip(num_nodes_incremental, num_nodes_layers, cur_conv_list, conv_kernel_size_list):
        # perm = solution_forward_list[idx + 1]
        # perm = solution_backward_list[len(solution_backward_list) - idx - 2]
        if idx + 1 <= num_layers / 2:
            perm = solution_forward_list[idx + 1]
        else:
            perm = solution_backward_list[num_layers - idx - 1]
        if torch.sum(perm).item() != perm.shape[0]:
            perm_is_complete = False
        assert 'bias' not in named_weight_list_0[idx][0]
        if len(named_weight_list_0[idx][1].shape) == 4:
            aligned_wt_0[idx] = (
                    perm.transpose(0, 1).to(torch.float64) @ aligned_wt_0[idx].to(torch.float64).permute(2, 3, 0,
                                                                                                         1)).permute(
                2, 3, 0, 1)
        else:
            # print( f'perm device is {perm.device}' )
            # print( f'aligned_wt_0 device is {aligned_wt_0[0].device}' )
            aligned_wt_0[idx] = perm.transpose(0, 1).to(torch.float64) @ aligned_wt_0[idx].to(torch.float64)
        idx += 1
        if idx >= num_layers:
            continue
        if 'bias' in named_weight_list_0[idx][0]:
            aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
            idx += 1
        if idx >= num_layers:
            continue
        # pre_conv = pre_conv_list[idx]
        if cur_conv and len(named_weight_list_0[idx][1].shape) == 2:
            aligned_wt_0[idx] = (
                    aligned_wt_0[idx].to(torch.float64).reshape(aligned_wt_0[idx].shape[0], pre_conv_out_channel,
                                                                -1).permute(0, 2, 1) @ perm.to(torch.float64)).permute(
                0, 2, 1).reshape(aligned_wt_0[idx].shape[0], -1)
        elif len(named_weight_list_0[idx][1].shape) == 4:
            aligned_wt_0[idx] = (
                    aligned_wt_0[idx].to(torch.float64).permute(2, 3, 0, 1) @ perm.to(torch.float64)).permute(2, 3,
                                                                                                              0, 1)
        else:
            aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
    assert idx == num_layers
    averaged_weights = []
    for idx, parameter in enumerate(networks[1].parameters()):
        averaged_weights.append((1 - args.ensemble_step) * aligned_wt_0[idx] + args.ensemble_step * parameter)
    # networks[0] = networks[0].cuda(args.gpu_id)
    # networks[1] = networks[1].cuda(args.gpu_id)
    return averaged_weights, perm_is_complete


def graph_matching_fusion(args, networks: list):
    '''
    the function use graph matching technique to align each layer in networks[0] along with
        networks[1], and return a list that contains the averaged aligned parameters, following
        the original order of parameters in model.parameters()
    the averaging weights are specified in [args.ensemble_step, 1-args.ensemble_step]
    '''
    '''
    count the number of nodes in network[0] and network[1], and store them
        as [n1] and [n2], respectively
    '''
    n1 = total_node_num(network=networks[0])
    n2 = total_node_num(network=networks[1])
    assert (n1 == n2)
    '''
    define affinity matrix
    '''
    affinity = torch.zeros([n1 * n2, n1 * n2], device=args.device)
    '''
    iterate through all the layers to calculate the pair-wise distances / affinities
    suppose the layer node numbers are:
        N1(inputs), N2, ..., N(l-1), Nl(outputs), then
    [num_nodes_incremental] = [ N1, N1+N2, ..., N1+N2+...+N(l-1) ]
    [num_nodes_layers]      = [ N2, N3,    ..., Nl               ]
    [pre_conv_list]         = [ False, conv(layer1), conv(layer2), ..., conv(layer(l-1)) ]
        it does not contain bias layers
    [conv_kernel_size_list] = [ kernel_size(1), ..., kernel_size(l) ]
    '''
    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    num_nodes_before = 0
    num_nodes_incremental = []
    num_nodes_layers = []
    pre_conv_list = []
    conv_kernel_size_list = []
    num_nodes_pre = 0
    num_nodes_cur = 0
    is_conv = False
    pre_conv = False
    pre_conv_kernel_size = None
    pre_conv_out_channel = 1
    is_bias = False
    is_final_bias = False
    perm_is_complete = True
    named_weight_list_0 = [named_parameter for named_parameter in networks[0].named_parameters()]
    for idx, ((_, fc_layer0_weight), (_, fc_layer1_weight)) in \
            enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):
        assert fc_layer0_weight.shape == fc_layer1_weight.shape
        layer_shape = fc_layer0_weight.shape
        num_nodes_cur = fc_layer0_weight.shape[0]
        if len(layer_shape) > 1:
            # if it's a fully-connected layer after a convolutional layer
            if pre_conv is True and len(layer_shape) == 2:
                num_nodes_pre = pre_conv_out_channel
            else:
                num_nodes_pre = fc_layer0_weight.shape[1]
        '''
        tell whether the layer is convolutional or fully-connected or bias
        '''
        # if is_bias is False:
        #     pre_conv = is_conv
        #     pre_conv_list.append( pre_conv )
        if len(named_weight_list_0[idx - 1][1].shape) == 1:
            pre_bias = True
        else:
            pre_bias = False
        if len(layer_shape) > 2:
            is_bias = False
            if pre_bias == False:
                pre_conv = is_conv
                pre_conv_list.append(pre_conv)
            is_conv = True
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            fc_layer0_weight_data = fc_layer0_weight.data.view(
                fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(
                fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
        elif len(layer_shape) == 2:
            is_bias = False
            if pre_bias == False:
                pre_conv = is_conv
                pre_conv_list.append(pre_conv)
            is_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data
        else:
            is_bias = True
            if pre_bias == False:
                pre_conv = is_conv
                pre_conv_list.append(pre_conv)
            is_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data
        '''
        if it's conv, update [pre_conv_out_channel]
        '''
        if is_conv:
            pre_conv_out_channel = num_nodes_cur
        '''
        tell whether it's the final bias layer
        '''
        if is_bias is True and idx == num_layers - 1:
            is_final_bias = True
        '''
        if it's the first layer, map the input nodes
        '''
        if idx == 0:
            for a in range(num_nodes_pre):
                affinity[(num_nodes_before + a) * n2 + num_nodes_before + a] \
                    [(num_nodes_before + a) * n2 + num_nodes_before + a] \
                    = 1
        '''
        if it's the final layer, map the output nodes
        '''
        if idx == num_layers - 2 and 'bias' in named_weight_list_0[idx + 1][0] or \
                idx == num_layers - 1 and 'bias' not in named_weight_list_0[idx][0]:
            for a in range(num_nodes_cur):
                affinity[(num_nodes_before + num_nodes_pre + a) * n2 + num_nodes_before + num_nodes_pre + a] \
                    [(num_nodes_before + num_nodes_pre + a) * n2 + num_nodes_before + num_nodes_pre + a] \
                    = 1
        '''
        calculate the edge-wise soft affinities between two models
        '''
        if is_bias is False:
            ground_metric = gm.Ground_Metric_GM(
                fc_layer0_weight_data, fc_layer1_weight_data, is_conv, is_bias,
                pre_conv, int(fc_layer0_weight_data.shape[1] / pre_conv_out_channel))
        else:
            ground_metric = gm.Ground_Metric_GM(
                fc_layer0_weight_data, fc_layer1_weight_data, is_conv, is_bias,
                pre_conv, 1)

        layer_affinity = ground_metric.process_soft_affinity(p=2)
        # print( f'is_conf = {is_conv}, fc layer shape is {fc_layer0_weight.shape}' )
        if is_bias is False:
            pre_conv_kernel_size = fc_layer0_weight.shape[3] if is_conv else None
            conv_kernel_size_list.append(pre_conv_kernel_size)
        '''
        copy the affinity values from [layer_affinity] to the corresponding positions
            in [affinity] matrix
        '''
        if is_bias is True and is_final_bias is False:
            for a in range(num_nodes_cur):
                for c in range(num_nodes_cur):
                    affinity[(num_nodes_before + a) * n2 + num_nodes_before + c] \
                        [(num_nodes_before + a) * n2 + num_nodes_before + c] \
                        = layer_affinity[a][c]
        elif is_final_bias is False:
            for a in range(num_nodes_pre):
                for b in range(num_nodes_cur):
                    affinity[
                    (num_nodes_before + a) * n2 + num_nodes_before:
                    (num_nodes_before + a) * n2 + num_nodes_before + num_nodes_pre,
                    (num_nodes_before + num_nodes_pre + b) * n2 + num_nodes_before + num_nodes_pre:
                    (num_nodes_before + num_nodes_pre + b) * n2 + num_nodes_before + num_nodes_pre + num_nodes_cur] \
                        = layer_affinity[a + b * num_nodes_pre].view(num_nodes_cur, num_nodes_pre).transpose(0, 1)
        '''
        update the total number of nodes that has already been considered in previous steps
        '''
        if is_bias is False:
            num_nodes_before += num_nodes_pre
            num_nodes_incremental.append(num_nodes_before)
            num_nodes_layers.append(num_nodes_cur)

    '''
    solve the quadratic assignment problem by calling gurobipy package
    '''
    solution = gb.gurobi_qap_solver(affinity, n1, n2, time_limit=300)

    # debug block begin (uncomment and unindent the following to debug)
    # torch. set_printoptions(profile="full")
    # print( f'affinity matrix is \n{affinity}' )
    # print( f'solution is \n{solution}' )
    # torch. set_printoptions(profile="default")
    # return
    # debug block end
    '''
    perform the alignment to network[0] according to the solution
    [idx] represents the index of layers, including 'bias' layers
    '''
    aligned_wt_0 = [parameter.data for name, parameter in named_weight_list_0]
    idx = 0
    num_layers = len(aligned_wt_0)
    '''
    for each iteration, the weight matrix between two layers (e.g. L_i and L_{i+1}) are considered
        [num_before] denotes N_1 + N_2 + ... + N_i
        [num_cur] denotes N_{i+1}
        [pre_conv] denotes whether L_i is convolutional
        [cur_kernel_size] denotes the kenrel_size of the current weight matrix
    '''
    for num_before, num_cur, pre_conv, cur_kernel_size in \
            zip(num_nodes_incremental, num_nodes_layers, pre_conv_list, conv_kernel_size_list):
        '''
        obtain permutation matrix according to the solution
        some preliminaries about permutation matrix:
            1.  firstly, we define a permuation function Pi: {1,...,M} --> {1,...,M}, so that
                1 is mapped to Pi(1), 2 is mapped to Pi(2), ..., M is mapped to Pi(M).
            2.  Then, we construct the corresponding M x M permutation matrix Perm by:
                Perm[i, j] = 1 if j == Pi(i) else 0
            3.  if we have a N x M matrix A, and we derive B = A @ Perm, then
                the [i]th column of A would become the [Pi(i)]th column of B
            4.  if we have a M x N matrix C, and we derive D = perm^T @ C, then
                the [i]th row of C would become the [Pi(i)]th row of D
        
        some structural information of the returned solution [solution]:
            1.  for the [i]th layer with ni nodes, and Ni nodes before,
                solution[Ni + a][Ni + b] = 1 if a is mapped to b else 0
            2.  if we define Perm_i = solution[Ni:Ni_ni][Ni:Ni+ni], then Perm_i is the 
                permutation matrix corresponding to the permutation function Pi, where
                the [i]th node in model 1 is mapped to [Pi(i)]th node in model 2
        
        the procedure to permutate the parameters:
            1.  given the permutation matrix upon layer i, permutate the columns of parameters
                between layer i and layer i+1
            2.  given the permutation matrix upon layer i, permutate the rows of parameters
                between layer i-1 and layer i
        '''
        perm = solution[num_before:num_before + num_cur, num_before:num_before + num_cur]
        if torch.sum(perm).item() != perm.shape[0]:
            perm_is_complete = False
        '''
        permutate the rows of parameters between previous layer and current layer
        if the current layer is convolutional:
            1.  permute the aligned weight by: 2-->0, 3-->1, 0-->2, 1-->3
            2.  multiply with the transpose of permutation matrix
            3.  restore the permutation
        else:
            directly multiply with the permutation matrix
        for detailed explanation for the operator '@', or __matmul__, or infix
            multiplication between matrices, see the link:
            https://www.python.org/dev/peps/pep-0465/
        '''
        assert 'bias' not in named_weight_list_0[idx][0]
        if len(named_weight_list_0[idx][1].shape) == 4:
            aligned_wt_0[idx] = (perm.transpose(0, 1).to(torch.float64) @ \
                                 aligned_wt_0[idx].to(torch.float64).permute(2, 3, 0, 1)) \
                .permute(2, 3, 0, 1)
        else:
            aligned_wt_0[idx] = perm.transpose(0, 1).to(torch.float64) @ aligned_wt_0[idx].to(torch.float64)
        idx += 1
        '''
        if the bias layer is present, then permuate the bias layer
        '''
        if idx >= num_layers:
            continue
        if 'bias' in named_weight_list_0[idx][0]:
            aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
            idx += 1
        '''
        permutate the columns of parameters between current layer and the next layer
        if the previous layer is convolutional and the current layer is fully-connected:
            1.  reshape the aligned weight to 
                [cur_num] x [pre_num / kernel_size_squared] x [kernel_size_squared]
            2.  permute the aligned weight so that dim 1 and dim 2 are switched
            3.  multiply the permutation matrix
            4.  permute the aligned weight so that dim 1 and dim 2 are restored
            5.  restore the shape of the aligned weight back to
                [cur_num] x [pre_num]
        else:
            directly multiply the permutation matrix
        '''
        if idx >= num_layers:
            continue
        if pre_conv and len(named_weight_list_0[idx][1].shape) == 2:
            aligned_wt_0[idx] = (aligned_wt_0[idx].to(torch.float64) \
                                 .reshape(aligned_wt_0[idx].shape[0], pre_conv_out_channel, -1) \
                                 .permute(0, 2, 1) \
                                 @ perm.to(torch.float64)) \
                .permute(0, 2, 1) \
                .reshape(aligned_wt_0[idx].shape[0], -1)
        elif len(named_weight_list_0[idx][1].shape) == 4:
            aligned_wt_0[idx] = (aligned_wt_0[idx].to(torch.float64) \
                                 .permute(2, 3, 0, 1) \
                                 @ perm.to(torch.float64)) \
                .permute(2, 3, 0, 1)
        else:
            aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
    assert idx == num_layers

    # debug block begin
    # for aligned_wt, (name, parameter) in zip( aligned_wt_0, networks[0].named_parameters() ):
    #     print( f'*the original weights named "{name}" are \n{parameter}\n*and the aligned \
    #         weights are \n{aligned_wt}' )
    # debug block end
    '''
    average the parameters of model 1 and model 2 according to the weights given by [args.ensemble_step, 1-args.ensemble_step], 
        then store the results in a list, and return the list
    '''
    averaged_weights = []
    for idx, parameter in enumerate(networks[1].parameters()):
        averaged_weights.append( ((1 - args.ensemble_step) * aligned_wt_0[idx] + args.ensemble_step * parameter).cuda(args.gpu_id) )
    return averaged_weights, perm_is_complete


def get_fused_model(args, networks: list, slice=False, sparse=False):
    '''
    the input [parameters] is a list consisting of tensors
    '''
    if not slice:
        parameters, perm_is_complete = graph_matching_fusion(args, networks)
    else:
        parameters, perm_is_complete = graph_matching_fusion_slice(args, networks, sparse)
    fused_model = model.get_model_from_name(args).cuda(args.gpu_id)
    state_dict = fused_model.state_dict()
    for idx, (key, _) in enumerate(state_dict.items()):
        state_dict[key] = parameters[idx]
    fused_model.load_state_dict(state_dict)
    return fused_model, perm_is_complete


if __name__ == "__main__":
    import torch.nn as nn
    import torch.nn.functional as F


    class dotdict(dict):
        """ dot.notation access to dictionary attributes """
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__


    args = dotdict({
        "weight": [0.5, 0.5],
        "model_name": "naivenet",
        "dataset": "mnist",
        "disable_bias": False,
        "width_ratio": 1,
        "num_hidden_nodes1": 20,
        "num_hidden_nodes2": 30,
        "num_hidden_nodes3": 10
    })
    '''
    define a very naive nueral network for testing purpose
    '''
    model1 = model.naive_net()
    model2 = model.naive_net()
    '''
    create two state_dict() instances to initialize two networks
    '''
    state_dict1 = {
        'lin1.weight': torch.tensor([[1, 2], [7, 8], [4, 5]]),
        'lin1.bias': torch.tensor([5, 6, 7]),
        'lin2.weight': torch.tensor([[1, 2, 3], [7, 8, 9]]),
        'lin2.bias': torch.tensor([4, 5])}
    state_dict2 = {
        'lin1.weight': torch.tensor([[2, 1], [4, 4], [7, 7]]),
        'lin1.bias': torch.tensor([4, 8, 6]),
        'lin2.weight': torch.tensor([[8, 7, 9], [2, 1, 3]]),
        'lin2.bias': torch.tensor([6, 3])}
    model1.load_state_dict(state_dict1)
    model2.load_state_dict(state_dict2)
    '''
    print two models to see that they are created as we wishes
    '''


    def print_model(model: nn.Module):
        for name, parameter in model.named_parameters():
            print(f'name is {name},\t parameter is \n\t{parameter}')


    # print( model1 )
    # print_model( model1 )
    # print( model2 )
    # print_model( model2 )
    # print('##########################################################')
    '''
    call the fusion function to check the affinity matrix and the solution
    '''
    # print( graph_matching_fusion( args, [model1, model2] ) )
    print(get_fused_model(args, [model1, model2]))
    # print('##########################################################')

    print('##########################################################')
    '''
    define a simple convolutional neural network
    '''
    args.model_name = 'naivecnn'
    model3 = model.naive_cnn()
    model4 = model.naive_cnn()
    '''
    create two state_dict() instances to initialize two networks
    '''
    state_dict3 = {
        'conv1.weight': torch.tensor([[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]]),
        'conv1.bias': torch.tensor([5, 6]),
        'fc1.weight': torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]]),
        'fc1.bias': torch.tensor([1, 2])}
    state_dict4 = {
        'conv1.weight': torch.tensor([[[[5, 6], [8, 7]]], [[[2, 1], [3, 4]]]]),
        'conv1.bias': torch.tensor([7, 4]),
        'fc1.weight': torch.tensor([[3, 4, 1, 2, 7, 8, 5, 6], [5, 7, 6, 8, 1, 3, 2, 4]]),
        'fc1.bias': torch.tensor([3, 1])}
    model3.load_state_dict(state_dict3)
    model4.load_state_dict(state_dict4)
    '''
    print two models to see that they are created as we wishes
    '''
    # print( model3 )
    # print_model( model3 )
    # print( model4 )
    # print_model( model4 )
    '''
    call the fusion function to check the affinity matrix and the solution
    '''
    # print('##########################################################')
    # print( graph_matching_fusion( args, [model3, model4] ) )
    print(get_fused_model(args, [model3, model4]))