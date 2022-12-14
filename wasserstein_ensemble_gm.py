from os import WUNTRACED
import ot
import torch
import numpy as np
from torch._C import device
import routines_gm as routines
from model_gm import get_model_from_name
import utils_gm
from ground_metric_original import GroundMetric
import math
import sys
# import compute_activations

def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c

def get_histogram(args, idx, cardinality, layer_name, activations=None, return_numpy = True, float64=False):
    '''
    return a vector that contains either uniform distribution or softmax of [activations]
        I don't understand what it's used for and why it's called "get_histogram"

    [cardinality] is the number of neurons in the current layer

    [args.unbalanced] = False indicates whether the histogram needs to be normalized

    [idx] is the index of MODEL -- NOT the index of the layer!
    '''
    if activations is None:
        # returns a uniform measure
        if not args.unbalanced:
            # Alexanderia
            # print("returns a uniform measure of cardinality: ", cardinality)
            return np.ones(cardinality)/cardinality
        else:
            return np.ones(cardinality)
    else:
        # return softmax over the activations raised to a temperature
        # layer_name is like 'fc1.weight', while activations only contains 'fc1'
        print(activations[idx].keys())
        unnormalized_weights = activations[idx][layer_name.split('.')[0]]
        print("For layer {},  shape of unnormalized weights is ".format(layer_name), unnormalized_weights.shape)
        unnormalized_weights = unnormalized_weights.squeeze()
        assert unnormalized_weights.shape[0] == cardinality

        if return_numpy:
            if float64:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy().astype(
                    np.float64)
            else:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy()
        else:
            return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0)


def get_wassersteinized_layers_modularized(args, networks, activations=None, eps=1e-7, test_loader=None):
    '''
    Two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*

    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
        basically the parameter [activations] is not used in this function
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    '''

    # simple_model_0, simple_model_1 = networks[0], networks[1]
    # simple_model_0 = get_trained_model(0, model='simplenet')
    # simple_model_1 = get_trained_model(1, model='simplenet')

    avg_aligned_layers = []
    # cumulative_T_var = None
    '''
    what is [T_var]?
    '''
    T_var = None
    # print(list(networks[0].parameters()))
    previous_layer_shape = None
    '''
    [ground_metric_object] is like a functional object
    '''
    ground_metric_object = GroundMetric(args)

    '''
    if [args.eval_aligned] == True, then the accuracy of the aligned model 0 would be evaluated
    '''
    if args.eval_aligned:
        model0_aligned_layers = []

    if args.gpu_id==-1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))

    '''
    [ zip(networks[0].parameters(), networks[1].parameters()) ] = zip(
        ( networks[0].parameters()[0], networks[0].parameters()[0] ),
        ( networks[0].parameters()[1], networks[1].parameters()[1] ),
        ...
        ( networks[0].parameters()[n], networks[1].parameters()[n] ) )

    [ list(zip(networks[0].parameters(), networks[1].parameters())) ] is a printable
        version of the above list
    
    [num_layers] therefore, records the number of layers that are common in two models
        Note that networks[i].parameters()[j] returns the tuple (name, weights) of [j]th 
        layer in the [i]th model
    '''
    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    for idx, ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) in \
            enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):
        '''
        in [i]th iteration,
        ( (layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight) )
            = ( networks[0].parameters()[i], networks[1].parameters()[i] )

        so  [layer0_name] = model[0]'s [i]th layer's name
            [fc_layer0_weight] = model[0]'s [i]th layer's weight value
            [layer1_name] = model[1]'s [i]th layer's name
            [fc_layer1_weight] = model[1]'s [i]th layer's weight value
        '''

        assert fc_layer0_weight.shape == fc_layer1_weight.shape
        # Alexanderia
        # print("Previous layer shape is ", previous_layer_shape)
        previous_layer_shape = fc_layer1_weight.shape

        '''
        [mu_cardinality] is the number of nodes in the current layer in model 0
        [nu_cardinality] is the number of nodes in the current layer in model 1
        '''
        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        # mu = np.ones(fc_layer0_weight.shape[0])/fc_layer0_weight.shape[0]
        # nu = np.ones(fc_layer1_weight.shape[0])/fc_layer1_weight.shape[0]

        layer_shape = fc_layer0_weight.shape
        '''
        for fully-conntected layers which are not convolutional, the weight is simply
            a two-dimensional matrix, so
            [layer_shape] = torch.Size([out_channels_num, in_channel_num])
            and [len(layer_shape)] = 2
        
        but for convolutional layers, the weight is 4-dimensional, which is 
            [layer_shape] = torch.Size([out_channel_num, in_channel_num, height, weight])
            therefore 
            [len(layer_shape)] = 4 > 2
        '''
        if len(layer_shape) > 2:
            is_conv = True
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            '''
            [fc_layer0_weight.data] returns the same tensor as [fc_layer0_weight] but
                without tracing for gradients
            
            [.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)] resizes
                the weight matrix so that the convolutional kernel in each edge is flattened.
                Now the weight is of the dimension 3
            '''
            fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
        else:
            is_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data

        if idx == 0:
            '''
            at the first iteration, need to initialize [is_conv] and [aligned_wt]

            also, 
            '''
            if is_conv:
                '''
                if it's convolutional network, further flatten the weight so that the kernels from the 
                    same in-channel is combined to a single vector. Then the distances between two 
                    such vectors is measured. Now the weight is of dimension 2
                '''
                M = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                                fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
                # M = cost_matrix(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                #                 fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
            else:
                '''
                if it's not convolutional, then directly measure the distance and it's done
                '''
                # print("layer data is ", fc_layer0_weight_data, fc_layer1_weight_data)
                M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)
                # M = cost_matrix(fc_layer0_weight, fc_layer1_weight)

            aligned_wt = fc_layer0_weight_data
        else:

            # Alexanderia
            # print("shape of layer: model 0", fc_layer0_weight_data.shape)
            # print("shape of layer: model 1", fc_layer1_weight_data.shape)
            # print("shape of previous transport map", T_var.shape)

            # aligned_wt = None, this caches the tensor and causes OOM
            '''
            the following if-and-else statement handles in-coming edges alignments
            '''
            if is_conv:
                '''
                T_var_conv: how is it derived?
                I guess [T_v] is two-dimensional?
                T_v is repeated for [kernel_size] times along a third dimension

                [permute] function permute the dimensions of the original tensor
                    in specific, if the tensor [x] has N dimensions, and
                    ` y = x.permute(d_1, d_2, ..., d_N)] `, 
                    then the tensor[y] is the result obtained by mapping dimension
                    [d_1] of [x] to [0], dimension [d_2] of [x] to [1], and so on.
                    For example, if [x] = [ [ [1, 2, 3 ],
                                              [4, 5, 6 ] ],
                                            [ [7, 8, 9 ],
                                              [10,11,12] ] ]
                    then y = x.permute(2, 0, 1)
                                        = [ [ [1, 4 ],
                                              [7, 10] ],
                                            [ [2, 5 ],
                                              [8, 11] ],
                                            [ [3, 6 ],
                                              [9, 12] ] ]

                bmm is batch matrix-matrix multiplication, and the two input matrices
                    must be both 3-D, and the first dimension must be the same.
                    ` torch.bmm(m1, m2) = [ m1[0] @ m2[0],
                                            m1[1] @ m2[1],
                                            ...
                                            m1[n] @ m2[n] ] `

                Therefore, since [fc_layer0_weight_data] is 
                    [
                        [ [kernel_1_1], [kernel_1_2], ..., [kernel_1_lcurrent] ],
                        [ [kernel_2_1], [kernel_2_2], ..., [kernel_2_lcurrent] ],
                        ...
                        [ [kernel_lprevious_1], [kernel_lprevious_2], ..., [kernel_lprevious_lcurrent] ]
                    ],
                    then [fc_layer0_weight_data.permute(2,0,1)] is
                    [
                        [K_0],
                        [K_1],
                        ...
                        [K_kernel_size_squared_minus_1]
                    ],
                    where [K_i] is the matrix
                    [
                        [ kernel_1_1[i], kernel_1_2[i], ..., kernel_1_lcurrent[i] ],
                        [ kernel_2_1[i], kernel_2_2[i], ..., kernel_2_lcurrent[i] ],
                        ...
                        [ kernel_lprevious_1[i], kernel_lprevious_2[i], ..., kernel_lprevious_lcurrent[i] ],
                    ]
                
                so the function ` torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv) calculates
                ` m_i = K_i @ T_var ` 
                and stores in a list [m_0, m_1, ..., m_kernel_squared_minus_1], which is [aligned_wt]

                finally, with a ` .permute(1, 2, 0) ` operation, it transforms back to the normal weight dimension,
                    which is
                    [
                        [ [kernel_1_1], [kernel_1_2], ..., [kernel_1_lcurrent] ],
                        [ [kernel_2_1], [kernel_2_2], ..., [kernel_2_lcurrent] ],
                        ...
                        [ [kernel_lprevious_1], [kernel_lprevious_2], ..., [kernel_lprevious_lcurrent] ]
                    ]
                
                therefore, essentially for convolutional network, OT applies entity-wise transformation
                '''
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)

                '''
                [aligned_wt.contiguous()] makes [view] function callable. Detailed reasons
                    involves the physical layouts in memory, and you don't need to care about that
                
                ` aligned_wt.contiguous().view(aligned_wt.shape[0], -1) ` step further flattens 
                    the weight so that the kernels from the same in-channel are combined into
                    a single vector
                '''
                M = ground_metric_object.process(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                )
            else:
                '''
                not a convolutional layer, so a fully-connected layer

                for fully connected layer directly after a convolutional layer, each pixel
                    in each channel is regarded as a neuron.
                    for example, if the previous layer has 6 channels, each outputs a matrix
                    with the size 5*5, then the in-channel for the current fully-connected layer
                    is 6*5*5, namely 
                    ` fc_layer0_weight.data.shape[1] = 6*5*5 `
                    but ` T_var.shape[0] = 6 `, therefore they would not be equal
                '''
                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    '''
                    the following [.view] operation combines the neurons in the same channel
                        into a vector and performs batch matrix multiplication
                    '''
                    # Handles the switch from convolutional layers to fc layers
                    fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened,
                        T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                else:
                    '''
                    if the previous layer is not convolutional, then directly do the matrix
                        multiplication and it's done
                    '''
                    # print("layer data (aligned) is ", aligned_wt, fc_layer1_weight_data)
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                # M = cost_matrix(aligned_wt, fc_layer1_weight)
                M = ground_metric_object.process(aligned_wt, fc_layer1_weight)
                # Alexanderia
                # print("ground metric is ", M)
            
            '''
            probably the last layer, namely the output layer has already been guaranteed
                to be consistent between two models, so the alignment could be skipped, 
                and the averaged weight could be directly returned
            '''
            if args.skip_last_layer and idx == (num_layers - 1):
                # Alexanderia
                # print("Simple averaging of last layer weights. NO transport map needs to be computed")
                if args.ensemble_step != 0.5:
                    avg_aligned_layers.append((1 - args.ensemble_step) * aligned_wt +
                                          args.ensemble_step * fc_layer1_weight)
                else:
                    avg_aligned_layers.append((aligned_wt + fc_layer1_weight)/2)
                return avg_aligned_layers

        '''
        if the last layer alignment is not skipped, now we need to:
            1. update the optimal transport matrix
            2. align the current layer weights
        '''

        '''
        determines the probability measures [mu] and [nu]

        [args.importance]: determines the criterion upon which the neurons in the previous
            layer should be weighted

        if [args.importance] is not specified or it's the last layer, directly set [mu] and [nu]
            to be the uniform measure
        
        else if [args.importance] is specified and it's not the last layer, 
        '''
        if args.importance is None or (idx == num_layers - 1):
            mu = get_histogram(args, 0, mu_cardinality, layer0_name)
            nu = get_histogram(args, 1, nu_cardinality, layer1_name)
        else:
            # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
            mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
            nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
            # Alexanderia
            # print(mu, nu)
            assert args.proper_marginals

        '''
        solves OT problem and calculate the OT metrix, which is stored in [T]

        [M] is the ground metric matrix

        [args.exact] determines whether exact OT is calculated instead of approximate OT
        '''
        cpuM = M.data.cpu().numpy()
        if args.exact:
            T = ot.emd(mu, nu, cpuM)
        else:
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)
        # T = ot.emd(mu, nu, log_cpuM)

        '''
        [T_var] is updated by [T]
        '''
        if args.gpu_id!=-1:
            T_var = torch.from_numpy(T).to(args.device).float()
        else:
            T_var = torch.from_numpy(T).to(args.device).float()

        # torch.set_printoptions(profile="full")
        # Alexanderia
        # print("the transport map is ", T_var)
        # torch.set_printoptions(profile="default")
        
        '''
        [args.correction] determines whether scaling correction is performed

        [args.proper_marginals] makes the marginals of OT matrix to be consider properly
            if False, marginals are just uniform distribution measures
            if True, marginals are authentical marginals
        '''
        if args.correction:
            if not args.proper_marginals:
                # think of it as m x 1, scaling weights for m linear combinations of points in X
                if args.gpu_id != -1:
                    # marginals = torch.mv(T_var.t(), torch.ones(T_var.shape[0]).cuda(args.gpu_id))  # T.t().shape[1] = T.shape[0]
                    marginals = torch.ones(T_var.shape[0]).to(args.device) / T_var.shape[0]
                else:
                    # marginals = torch.mv(T_var.t(),
                    #                      torch.ones(T_var.shape[0]))  # T.t().shape[1] = T.shape[0]
                    marginals = torch.ones(T_var.shape[0]).to(args.device) / T_var.shape[0]
                marginals = torch.diag(1.0/(marginals + eps))  # take inverse
                T_var = torch.matmul(T_var, marginals)
            else:
                '''
                [.t()] method returns the transpose
                '''
                # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
                marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

                marginals = (1 / (marginals_beta + eps))
                # Alexanderia
                # print("shape of inverse marginals beta is ", marginals_beta.shape)
                # Alexanderia
                # print("inverse marginals beta is ", marginals_beta)
                '''
                * operation between matrices is entity-wise!!!
                '''
                T_var = T_var * marginals
                # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
                # this should all be ones, and number equal to number of neurons in 2nd model
                # Alexanderia
                # print(T_var.sum(dim=0))
                # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

        if args.debug:
            if idx == (num_layers - 1):
                print("there goes the last transport map: \n ", T_var)
            else:
                print("there goes the transport map at layer {}: \n ".format(idx), T_var)

            print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))

        # Alenxanderia
        # print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))
        # Alenxanderia
        # print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
        setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

        '''
        if [past_correction] is True, would align the previously obtained weight (whose in-coming
            edges has been already aligned)
        
        otherwise if [past_correction] is False, align the original weight

        the finally aligned weights are stored in [t_fc0_model]
        '''
        if args.past_correction:
            # Alenxanderia
            # print("this is past correction for weight mode")
            # Alenxanderia
            # print("Shape of aligned wt is ", aligned_wt.shape)
            # Alenxanderia
            # print("Shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)
            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))

        '''
        average the aligned weights of model 0 and the original weights of model 1
        '''
        # Average the weights of aligned first layers
        if args.ensemble_step != 0.5:
            geometric_fc = ((1-args.ensemble_step) * t_fc0_model +
                            args.ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
        else:
            geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))/2
        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
        avg_aligned_layers.append(geometric_fc)

        # get the performance of the model 0 aligned with respect to the model 1
        if args.eval_aligned:
            if is_conv and layer_shape != t_fc0_model.shape:
                t_fc0_model = t_fc0_model.view(layer_shape)
            model0_aligned_layers.append(t_fc0_model)
            _, acc = update_model(args, networks[0], model0_aligned_layers, test=True,
                                  test_loader=test_loader, idx=0)
            # Alexanderia
            # print("For layer idx {}, accuracy of the updated model is {}".format(idx, acc))
            setattr(args, 'model0_aligned_acc_layer_{}'.format(str(idx)), acc)
            if idx == (num_layers - 1):
                setattr(args, 'model0_aligned_acc', acc)

    return avg_aligned_layers


def print_stats(arr, nick=""):
    print(nick)
    print("summary stats are: \n max: {}, mean: {}, min: {}, median: {}, std: {} \n".format(
        arr.max(), arr.mean(), arr.min(), np.median(arr), arr.std()
    ))

def get_activation_distance_stats(activations_0, activations_1, layer_name=""):
    if layer_name != "":
        print("In layer {}: getting activation distance statistics".format(layer_name))
    M = cost_matrix(activations_0, activations_1) ** (1/2)
    mean_dists =  torch.mean(M, dim=-1)
    max_dists = torch.max(M, dim=-1)[0]
    min_dists = torch.min(M, dim=-1)[0]
    std_dists = torch.std(M, dim=-1)

    print("Statistics of the distance from neurons of layer 1 (averaged across nodes of layer 0): \n")
    print("Max : {}, Mean : {}, Min : {}, Std: {}".format(torch.mean(max_dists), torch.mean(mean_dists), torch.mean(min_dists), torch.mean(std_dists)))

def ot_weight_align( w:torch.tensor, anchor:torch.tensor, T_var_pre:torch.tensor, device ):
    '''
    [w] and [anchor] are parameters for only a layer.
    [T_var_pre] is the optimal transport solution for the previous layer. 
        If the current layer is the first layer, then this variable is None
    
    this function aligns the weight [w] with [anchor]
    returns the aligned [aligned_w]
    all the operations are done on cpu, so is the returned result
    '''
    from utils_gm import Namespace
    import ot
    layer_shape = w.shape
    args = Namespace( 
        ground_metric='euclidean', 
        ground_metric_normalize='none', 
        reg=0.01, 
        debug=False, 
        clip_max=5, 
        clip_min=0, 
        activation_histograms=True, 
        dist_normalize=True, 
        act_num_samples=200, 
        geom_ensemble_type='acts', 
        normalize_wts=False, 
        clip_gm=False, 
        device=device,
        importance=None,
        unbalanced=False,
        ground_metric_eff=False    )
    ground_matrix_object = GroundMetric( args )
    if len( layer_shape ) > 2:
        is_conv = True
        w = w.view( layer_shape[0], layer_shape[1], -1 )
        anchor = anchor.view( layer_shape[0], layer_shape[1], -1 )
        if T_var_pre is not None:
            T_var_conv = T_var_pre.unsqueeze( 0 ).repeat( w.shape[2], 1, 1 )
            aligned_w = torch.bmm( w.permute( 2, 0, 1 ), T_var_conv ).permute( 1, 2, 0 )
        else:
            aligned_w = w
        M = ground_matrix_object.process( 
            aligned_w.contiguous().view( aligned_w.shape[0], -1 ), 
            anchor.view( anchor.shape[0], -1 ) )
        
    else:
        is_conv = False
        if T_var_pre is not None:
            if layer_shape[1] != T_var_pre.shape[0]:
                w_unflattened = w.view(w.shape[0], T_var_pre.shape[0], -1).permute(2, 0, 1)
                aligned_w = torch.bmm(
                    w_unflattened,
                    T_var_pre.unsqueeze(0).repeat(w_unflattened.shape[0], 1, 1)
                ).permute(1, 2, 0)
                aligned_w = aligned_w.contiguous().view(aligned_w.shape[0], -1)
            else:
                aligned_w = torch.matmul( w, T_var_pre )
        else:
            aligned_w = w
        M = ground_matrix_object.process( aligned_w, anchor )
    # mu = _get_neuron_importance_histogram(args, w, is_conv)
    mu = get_histogram(args, 0, w.shape[0], None)
    # nu = _get_neuron_importance_histogram(args, w, is_conv)
    nu = get_histogram(args, 1, w.shape[0], None)
    cpuM = M.data.cpu().numpy()
    T = ot.emd(mu, nu, cpuM)
    T_var_current = torch.from_numpy(T).float()
    marginals = torch.ones(T_var_current.shape[0]) / T_var_current.shape[0]
    marginals = torch.diag(1.0/(marginals + 1e-7))  # take inverse

    # print(f'device for T_var_current.t() is {T_var_current.t().device}')
    # print(f'device for aligned_w.contiguous().view(aligned_w.shape[0], -1) is {aligned_w.contiguous().view(aligned_w.shape[0], -1).device}')

    aligned_w = torch.matmul(T_var_current.t(), aligned_w.contiguous().view(aligned_w.shape[0], -1))
    aligned_w = aligned_w.view( layer_shape )
    return aligned_w, T_var_current


def update_model(args, model, new_params, test=False, test_loader=None, reversed=False, idx=-1):

    updated_model = get_model_from_name(args, idx=idx)
    if args.gpu_id != -1:
        updated_model = updated_model.cuda(args.gpu_id)

    layer_idx = 0
    model_state_dict = model.state_dict()

    # Alexanderia
    # print("len of model_state_dict is ", len(model_state_dict.items()))
    print("len of new_params is ", len(new_params))

    for key, value in model_state_dict.items():
        print("updated parameters for layer ", key)
        model_state_dict[key] = new_params[layer_idx]
        layer_idx += 1
        if layer_idx == len(new_params):
            break


    updated_model.load_state_dict(model_state_dict)

    if test:
        log_dict = {}
        log_dict['test_losses'] = []
        final_acc = routines.test(args, updated_model, test_loader, log_dict)
        print("accuracy after update is ", final_acc)
    else:
         final_acc = None

    return updated_model, final_acc

def _check_activation_sizes(args, acts0, acts1):
    if args.width_ratio == 1:
        return acts0.shape == acts1.shape
    else:
        return acts0.shape[-1]/acts1.shape[-1] == args.width_ratio

def process_activations(args, activations, layer0_name, layer1_name):
    activations_0 = activations[0][layer0_name.replace('.' + layer0_name.split('.')[-1], '')].squeeze(1)
    activations_1 = activations[1][layer1_name.replace('.' + layer1_name.split('.')[-1], '')].squeeze(1)

    # assert activations_0.shape == activations_1.shape
    _check_activation_sizes(args, activations_0, activations_1)

    if args.same_model != -1:
        # sanity check when averaging the same model (with value being the model index)
        assert (activations_0 == activations_1).all()
        print("Are the activations the same? ", (activations_0 == activations_1).all())

    if len(activations_0.shape) == 2:
        activations_0 = activations_0.t()
        activations_1 = activations_1.t()
    elif len(activations_0.shape) > 2:
        reorder_dim = [l for l in range(1, len(activations_0.shape))]
        reorder_dim.append(0)
        print("reorder_dim is ", reorder_dim)
        activations_0 = activations_0.permute(*reorder_dim).contiguous()
        activations_1 = activations_1.permute(*reorder_dim).contiguous()

    return activations_0, activations_1

def _reduce_layer_name(layer_name):
    # print("layer0_name is ", layer0_name) It was features.0.weight
    # previous way assumed only one dot, so now I replace the stuff after last dot
    return layer_name.replace('.' + layer_name.split('.')[-1], '')

def _get_layer_weights(layer_weight, is_conv):
    if is_conv:
        # For convolutional layers, it is (#out_channels, #in_channels, height, width)
        layer_weight_data = layer_weight.data.view(layer_weight.shape[0], layer_weight.shape[1], -1)
    else:
        layer_weight_data = layer_weight.data

    return layer_weight_data

def _process_ground_metric_from_acts(args, is_conv, ground_metric_object, activations):
    print("inside refactored")
    if is_conv:
        if not args.gromov:
            M0 = ground_metric_object.process(activations[0].view(activations[0].shape[0], -1),
                                             activations[1].view(activations[1].shape[0], -1))
        else:
            M0 = ground_metric_object.process(activations[0].view(activations[0].shape[0], -1),
                                              activations[0].view(activations[0].shape[0], -1))
            M1 = ground_metric_object.process(activations[1].view(activations[1].shape[0], -1),
                                              activations[1].view(activations[1].shape[0], -1))

        print("# of ground metric features is ", (activations[0].view(activations[0].shape[0], -1)).shape[1])
    else:
        if not args.gromov:
            M0 = ground_metric_object.process(activations[0], activations[1])
        else:
            M0 = ground_metric_object.process(activations[0], activations[0])
            M1 = ground_metric_object.process(activations[1], activations[1])

    if args.gromov:
        return M0, M1
    else:
        return M0, None


def _custom_sinkhorn(args, mu, nu, cpuM):
    if not args.unbalanced:
        if args.sinkhorn_type == 'normal':
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == 'stabilized':
            T = ot.bregman.sinkhorn_stabilized(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == 'epsilon':
            T = ot.bregman.sinkhorn_epsilon_scaling(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == 'gpu':
            T, _ = utils.sinkhorn_loss(cpuM, mu, nu, gpu_id=args.gpu_id, epsilon=args.reg, return_tmap=True)
        else:
            raise NotImplementedError
    else:
        T = ot.unbalanced.sinkhorn_knopp_unbalanced(mu, nu, cpuM, reg=args.reg, reg_m=args.reg_m)
    return T


def _sanity_check_tmap(T):
    if not math.isclose(np.sum(T), 1.0, abs_tol=1e-7):
        print("Sum of transport map is ", np.sum(T))
        raise Exception('NAN inside Transport MAP. Most likely due to large ground metric values')

def _get_updated_acts_v0(args, layer_shape, aligned_wt, model0_aligned_layers, networks, test_loader, layer_names):
    '''
    Return the updated activations of the 0th model with respect to the other one.

    :param args:
    :param layer_shape:
    :param aligned_wt:
    :param model0_aligned_layers:
    :param networks:
    :param test_loader:
    :param layer_names:
    :return:
    '''
    if layer_shape != aligned_wt.shape:
        updated_aligned_wt = aligned_wt.view(layer_shape)
    else:
        updated_aligned_wt = aligned_wt

    updated_model0, _ = update_model(args, networks[0], model0_aligned_layers + [updated_aligned_wt], test=True,
                                     test_loader=test_loader, idx=0)
    updated_activations = utils.get_model_activations(args, [updated_model0, networks[1]],
                                                      config=args.config,
                                                      layer_name=_reduce_layer_name(layer_names[0]), selective=True)

    updated_activations_0, updated_activations_1 = process_activations(args, updated_activations,
                                                                       layer_names[0], layer_names[1])
    return updated_activations_0, updated_activations_1

def _get_updated_acts_v1(args, networks, test_loader, layer_names):
    '''
    Return the updated activations of the 0th model with respect to the other one.

    :param args:
    :param layer_shape:
    :param aligned_wt:
    :param model0_aligned_layers:
    :param networks:
    :param test_loader:
    :param layer_names:
    :return:
    '''
    updated_activations = utils.get_model_activations(args, networks,
                                                      config=args.config)

    updated_activations_0, updated_activations_1 = process_activations(args, updated_activations,
                                                                       layer_names[0], layer_names[1])
    return updated_activations_0, updated_activations_1

def _check_layer_sizes(args, layer_idx, shape1, shape2, num_layers):
    if args.width_ratio == 1:
        return shape1 == shape2
    else:
        if args.dataset == 'mnist':
            if layer_idx == 0:
                return shape1[-1] == shape2[-1] and (shape1[0]/shape2[0]) == args.width_ratio
            elif layer_idx == (num_layers -1):
                return (shape1[-1]/shape2[-1]) == args.width_ratio and shape1[0] == shape2[0]
            else:
                ans = True
                for ix in range(len(shape1)):
                    ans = ans and shape1[ix]/shape2[ix] == args.width_ratio
                return ans
        elif args.dataset[0:7] == 'Cifar10':
            assert args.second_model_name is not None
            if layer_idx == 0 or layer_idx == (num_layers -1):
                return shape1 == shape2
            else:
                if (not args.reverse and layer_idx == (num_layers-2)) or (args.reverse and layer_idx == 1):
                    return (shape1[1] / shape2[1]) == args.width_ratio
                else:
                    return (shape1[0]/shape2[0]) == args.width_ratio


def _compute_marginals(args, T_var, device, eps=1e-7):
    if args.correction:
        if not args.proper_marginals:
            # think of it as m x 1, scaling weights for m linear combinations of points in X
            marginals = torch.ones(T_var.shape)
            if args.gpu_id != -1:
                marginals = marginals.cuda(args.gpu_id)

            marginals = torch.matmul(T_var, marginals)
            marginals = 1 / (marginals + eps)
            print("marginals are ", marginals)

            T_var = T_var * marginals

        else:
            # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
            marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

            marginals = (1 / (marginals_beta + eps))
            print("shape of inverse marginals beta is ", marginals_beta.shape)
            print("inverse marginals beta is ", marginals_beta)

            T_var = T_var * marginals
            # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
            # this should all be ones, and number equal to number of neurons in 2nd model
            print(T_var.sum(dim=0))
            # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

        print("T_var after correction ", T_var)
        print("T_var stats: max {}, min {}, mean {}, std {} ".format(T_var.max(), T_var.min(), T_var.mean(),
                                                                     T_var.std()))
    else:
        marginals = None

    return T_var, marginals

def _get_current_layer_transport_map(args, mu, nu, M0, M1, idx, layer_shape, eps=1e-7, layer_name=None):

    if not args.gromov:
        cpuM = M0.data.cpu().numpy()
        if args.exact:
            T = ot.emd(mu, nu, cpuM)
        else:
            T = _custom_sinkhorn(args, mu, nu, cpuM)

        if args.print_distances:
            ot_cost = np.multiply(T, cpuM).sum()
            print(f'At layer idx {idx} and shape {layer_shape}, the OT cost is ', ot_cost)
            if layer_name is not None:
                setattr(args, f'{layer_name}_layer_{idx}_cost', ot_cost)
            else:
                setattr(args, f'layer_{idx}_cost', ot_cost)
    else:
        cpuM0 = M0.data.cpu().numpy()
        cpuM1 = M1.data.cpu().numpy()

        assert not args.exact
        T = ot.gromov.entropic_gromov_wasserstein(cpuM0, cpuM1, mu, nu, loss_fun=args.gromov_loss, epsilon=args.reg)

    if not args.unbalanced:
        _sanity_check_tmap(T)

    if args.gpu_id != -1:
        T_var = torch.from_numpy(T).cuda(args.gpu_id).float()
    else:
        T_var = torch.from_numpy(T).float()

    if args.tmap_stats:
        print(
        "Tmap stats (before correction) \n: For layer {}, frobenius norm from the joe's transport map is {}".format(
            layer0_name, torch.norm(T_var - torch.ones_like(T_var) / torch.numel(T_var), p='fro')
        ))

    print("shape of T_var is ", T_var.shape)
    print("T_var before correction ", T_var)

    return T_var

def _get_neuron_importance_histogram(args, layer_weight, is_conv, eps=1e-9):
    '''
    this function calculates norm of the vectors of the parameters of previous layer neurons
        for example, suppose the returned list is [hist], then
        ` hist[i] = norm of out-parameters for ith neuron in the previous layer `

    [args.importance] determines which norm to calculate

    [args.unbalanced] = False makes the returned list normalized
    '''
    print('shape of layer_weight is ', layer_weight.shape)
    if is_conv:
        '''
        flatten so that kernels in the same in-channel are combined into a vector
        '''
        layer = layer_weight.contiguous().view(layer_weight.shape[0], -1).cpu().numpy()
    else:
        layer = layer_weight.cpu().numpy()
    
    if args.importance == 'l1':
        importance_hist = np.linalg.norm(layer, ord=1, axis=-1).astype(
                    np.float64) + eps
    elif args.importance == 'l2':
        importance_hist = np.linalg.norm(layer, ord=2, axis=-1).astype(
                    np.float64) + eps
    else:
        raise NotImplementedError

    if not args.unbalanced:
        importance_hist = (importance_hist/importance_hist.sum())
        print('sum of importance hist is ', importance_hist.sum())
    # assert importance_hist.sum() == 1.0
    return importance_hist

def get_acts_wassersteinized_layers_modularized(args, networks, activations, eps=1e-7, train_loader=None, test_loader=None):
    '''
    Average based on the activation vector over data samples. Obtain the transport map,
    and then based on which align the nodes and average the weights!
    Like before: two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*
    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    '''


    avg_aligned_layers = []
    T_var = None
    if args.handle_skips:
        skip_T_var = None
        skip_T_var_idx = -1
        residual_T_var = None
        residual_T_var_idx = -1

    marginals_beta = None
    # print(list(networks[0].parameters()))
    previous_layer_shape = None
    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    ground_metric_object = GroundMetric(args)

    if args.update_acts or args.eval_aligned:
        model0_aligned_layers = []

    if args.gpu_id==-1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))

    networks_named_params = list(zip(networks[0].named_parameters(), networks[1].named_parameters()))
    idx = 0
    incoming_layer_aligned = True # for input
    while idx < num_layers:
        ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) = networks_named_params[idx]
    # for idx,  in \
    #         enumerate(zip(network0_named_params, network1_named_params)):
        print("\n--------------- At layer index {} ------------- \n ".format(idx))
        # layer shape is out x in
        # assert fc_layer0_weight.shape == fc_layer1_weight.shape
        assert _check_layer_sizes(args, idx, fc_layer0_weight.shape, fc_layer1_weight.shape, num_layers)
        print("Previous layer shape is ", previous_layer_shape)
        previous_layer_shape = fc_layer1_weight.shape

        # will have shape layer_size x act_num_samples
        layer0_name_reduced = _reduce_layer_name(layer0_name)
        layer1_name_reduced = _reduce_layer_name(layer1_name)

        print("let's see the difference in layer names", layer0_name.replace('.' + layer0_name.split('.')[-1], ''), layer0_name_reduced)
        print(activations[0][layer0_name.replace('.' + layer0_name.split('.')[-1], '')].shape, 'shape of activations generally')
        # for conv layer I need to make the act_num_samples dimension the last one, but it has the intermediate dimensions for
        # height and width of channels, so that won't work.
        # So convert (num_samples, layer_size, ht, wt) -> (layer_size, ht, wt, num_samples)

        activations_0, activations_1 = process_activations(args, activations, layer0_name, layer1_name)

        # print("activations for 1st model are ", activations_0)
        # print("activations for 2nd model are ", activations_1)


        assert activations_0.shape[0] == fc_layer0_weight.shape[0]
        assert activations_1.shape[0] == fc_layer1_weight.shape[0]

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        get_activation_distance_stats(activations_0, activations_1, layer0_name)

        layer0_shape = fc_layer0_weight.shape
        layer_shape = fc_layer1_weight.shape
        if len(layer_shape) > 2:
            is_conv = True
        else:
            is_conv = False

        fc_layer0_weight_data = _get_layer_weights(fc_layer0_weight, is_conv)
        fc_layer1_weight_data = _get_layer_weights(fc_layer1_weight, is_conv)

        if idx == 0 or incoming_layer_aligned:
            aligned_wt = fc_layer0_weight_data

        else:

            print("shape of layer: model 0", fc_layer0_weight_data.shape)
            print("shape of layer: model 1", fc_layer1_weight_data.shape)

            print("shape of activations: model 0", activations_0.shape)
            print("shape of activations: model 1", activations_1.shape)


            print("shape of previous transport map", T_var.shape)

            # aligned_wt = None, this caches the tensor and causes OOM
            if is_conv:
                if args.handle_skips:
                    assert len(layer0_shape) == 4
                    # save skip_level transport map if there is block ahead
                    if layer0_shape[1] != layer0_shape[0]:
                        if not (layer0_shape[2] == 1 and layer0_shape[3] == 1):
                            print(f'saved skip T_var at layer {idx} with shape {layer0_shape}')
                            skip_T_var = T_var.clone()
                            skip_T_var_idx = idx
                        else:
                            print(
                                f'utilizing skip T_var saved from layer layer {skip_T_var_idx} with shape {skip_T_var.shape}')
                            # if it's a shortcut (128, 64, 1, 1)
                            residual_T_var = T_var.clone()
                            residual_T_var_idx = idx  # use this after the skip
                            T_var = skip_T_var
                        print("shape of previous transport map now is", T_var.shape)
                    else:
                        if residual_T_var is not None and (residual_T_var_idx == (idx - 1)):
                            T_var = (T_var + residual_T_var) / 2
                            print("averaging multiple T_var's")
                        else:
                            print("doing nothing for skips")
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)

            else:
                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    # Handles the switch from convolutional layers to fc layers
                    # checks if the input has been reshaped
                    fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0],
                                                                       -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened,
                        T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                else:
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)


        #### Refactored ####

            if args.update_acts:
                assert args.second_model_name is None
                activations_0, activations_1 = _get_updated_acts_v0(args, layer_shape, aligned_wt,
                                                                    model0_aligned_layers, networks,
                                                                    test_loader, [layer0_name, layer1_name])

        if args.importance is None or (idx == num_layers -1):
            mu = get_histogram(args, 0, mu_cardinality, layer0_name)
            nu = get_histogram(args, 1, nu_cardinality, layer1_name)
        else:
            # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
            mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
            nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
            print(mu, nu)
            assert args.proper_marginals

        if args.act_bug:
            # bug from before (didn't change the activation part)
            # only for reproducing results from previous version
            M0 = ground_metric_object.process(
                aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
            )
        else:
            # debugged part
            print("Refactored ground metric calc")
            M0, M1 = _process_ground_metric_from_acts(args, is_conv, ground_metric_object,
                                                      [activations_0, activations_1])

            print("# of ground metric features in 0 is  ", (activations_0.view(activations_0.shape[0], -1)).shape[1])
            print("# of ground metric features in 1 is  ", (activations_1.view(activations_1.shape[0], -1)).shape[1])

        if args.debug and not args.gromov:
            # bug from before (didn't change the activation part)
            M_old = ground_metric_object.process(
                aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
            )
            print("Frobenius norm of old (i.e. bug involving wts) and new are ",
                  torch.norm(M_old, 'fro'), torch.norm(M0, 'fro'))
            print("Frobenius norm of difference between ground metric wrt old ",
                  torch.norm(M0 - M_old, 'fro') / torch.norm(M_old, 'fro'))

            print("ground metric old (i.e. bug involving wts) is ", M_old)
            print("ground metric new is ", M0)

        ####################

        if args.same_model!=-1:
            print("Checking ground metric matrix in case of same models")
            if not args.gromov:
                print(M0)
            else:
                print(M0, M1)

        if args.skip_last_layer and idx == (num_layers - 1):

            if args.skip_last_layer_type == 'average':
                print("Simple averaging of last layer weights. NO transport map needs to be computed")
                if args.ensemble_step != 0.5:
                    print("taking baby steps (even in skip) ! ")
                    avg_aligned_layers.append((1-args.ensemble_step) * aligned_wt +
                                              args.ensemble_step * fc_layer1_weight)
                else:
                    avg_aligned_layers.append(((aligned_wt + fc_layer1_weight)/2))
            elif args.skip_last_layer_type == 'second':
                print("Just giving the weights of the second model. NO transport map needs to be computed")
                avg_aligned_layers.append(fc_layer1_weight)

            return avg_aligned_layers

        print("ground metric (m0) is ", M0)

        T_var = _get_current_layer_transport_map(args, mu, nu, M0, M1, idx=idx, layer_shape=layer_shape, eps=eps, layer_name=layer0_name)

        T_var, marginals = _compute_marginals(args, T_var, device, eps=eps)

        if args.debug:
            if idx == (num_layers - 1):
                print("there goes the last transport map: \n ", T_var)
                print("and before marginals it is ", T_var/marginals)
            else:
                print("there goes the transport map at layer {}: \n ".format(idx), T_var)

        print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))
        print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
        setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

        if args.past_correction:
            print("Shape of aligned wt is ", aligned_wt.shape)
            print("Shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)

            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))

        # Average the weights of aligned first layers
        if args.ensemble_step != 0.5:
            print("taking baby steps! ")
            geometric_fc = (1 - args.ensemble_step) * t_fc0_model + \
                           args.ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
        else:
            geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)) / 2
        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
        avg_aligned_layers.append(geometric_fc)


        # print("The averaged parameters are :", geometric_fc)
        # print("The model0 and model1 parameters were :", fc_layer0_weight.data, fc_layer1_weight.data)

        if args.update_acts or args.eval_aligned:
            assert args.second_model_name is None
            # the thing is that there might be conv layers or other more intricate layers
            # hence there is no point in having them here
            # so instead call the compute_activations script and pass it the model0 aligned layers
            # and also the aligned weight computed (which has been aligned via the prev T map, i.e. incoming edges).
            if is_conv and layer_shape != t_fc0_model.shape:
                t_fc0_model = t_fc0_model.view(layer_shape)
            model0_aligned_layers.append(t_fc0_model)
            _, acc = update_model(args, networks[0], model0_aligned_layers, test=True,
                                  test_loader=test_loader, idx=0)
            print("For layer idx {}, accuracy of the updated model is {}".format(idx, acc))
            setattr(args, 'model0_aligned_acc_layer_{}'.format(str(idx)), acc)
            if idx == (num_layers - 1):
                setattr(args, 'model0_aligned_acc', acc)

        incoming_layer_aligned = False
        next_aligned_wt_reshaped = None

        # remove cached variables to prevent out of memory
        activations_0 = None
        activations_1 = None
        mu = None
        nu = None
        fc_layer0_weight_data = None
        fc_layer1_weight_data = None
        M0 = None
        M1 = None
        cpuM = None

        idx += 1
    return avg_aligned_layers

def get_network_from_param_list(args, param_list, test_loader):

    # Alexanderia
    # print("using independent method")
    new_network = get_model_from_name(args, idx=1)
    if args.gpu_id != -1:
        new_network = new_network.cuda(args.gpu_id)

    # check the test performance of the network before
    log_dict = {}
    log_dict['test_losses'] = []
    # Alexanderia
    # routines.test(args, new_network, test_loader, log_dict)

    # set the weights of the new network
    # print("before", new_network.state_dict())
    # Alexanderia
    # print("len of model parameters and avg aligned layers is ", len(list(new_network.parameters())),
        #   len(param_list))
    assert len(list(new_network.parameters())) == len(param_list)

    layer_idx = 0
    model_state_dict = new_network.state_dict()

    # Alexanderia
    # print("len of model_state_dict is ", len(model_state_dict.items()))
    # Alexanderia
    # print("len of param_list is ", len(param_list))

    for key, value in model_state_dict.items():
        model_state_dict[key] = param_list[layer_idx]
        layer_idx += 1

    new_network.load_state_dict(model_state_dict)

    # check the test performance of the network after
    log_dict = {}
    log_dict['test_losses'] = []
    acc = routines.test(args, new_network, test_loader, log_dict)

    return acc, new_network

def geometric_ensembling_modularized(args, networks, train_loader, test_loader, activations=None):
    '''
    do geometric (namely OT-based) fusion to the models contained in [networks] and return the 
        model accuracy as well as the fused model
    
    if [geom_ensemble_type] == 'wts', then do the fusion with ground metric being the weights

    else if [geom_ensemble_type] == 'acts', then do the fusion with ground metric being the 
        activations specified in [activations]
    
    the [train_loader] and [test_loader] provides the data for training and testing, respectively
    '''
    if args.geom_ensemble_type == 'wts':
        avg_aligned_layers = get_wassersteinized_layers_modularized(args, networks, activations, test_loader=test_loader)
    elif args.geom_ensemble_type == 'acts':
        avg_aligned_layers = get_acts_wassersteinized_layers_modularized(args, networks, activations, train_loader=train_loader, test_loader=test_loader)
        
    return get_network_from_param_list(args, avg_aligned_layers, test_loader)

