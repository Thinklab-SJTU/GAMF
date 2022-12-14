import ground_metric_gm as gm
import gurobi_qap as gb
import numpy as np
import scipy.sparse as ss
import torch
import random
import sys
from hungarian import hungarian
from sinkhorn import Sinkhorn
import heapq

import os

# the following imports are from original OT package.
# Please change the directories based on the file distributions your computer
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../")))
import model_gm as model
# import ground_metric as gm
from ground_metric_ import GroundMetric
# import routines


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


def total_node_num_parameters(parameters: dict):
    '''
    count the total number of nodes in the network [network]
    '''
    num_nodes = 0
    for idx, (name, parameters) in enumerate(parameters.items()):
        if 'bias' in name:
            continue
        if idx == 0:
            num_nodes += parameters.shape[1]
        num_nodes += parameters.shape[0]
    return num_nodes


def get_lap_affinity_new(args, w0, w1, sol, is_conv, is_bias, pre_conv, channel, position=0):
    print(is_conv, is_bias, pre_conv, channel, position)
    if len(w0.shape) == 2:
        if w0.shape[1] != sol.shape[0]:
            if position == 0:
                weight_pre_0 = w0
                fc_layer0 = weight_pre_0.view(weight_pre_0.shape[0], sol.shape[0], -1).permute(
                    2, 0, 1)
                aligned_wt = torch.bmm(
                    fc_layer0, sol.unsqueeze(0).repeat(fc_layer0.shape[0], 1, 1)
                ).permute(1, 2, 0)
                weight_pre_0 = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                weight_pre_1 = w1
            else:
                num_cur_node = position
                weight_pre_0 = w0
                weight_pre_1 = w1
                fc_layer0 = weight_pre_0.view(sol.shape[0], num_cur_node, -1).permute(
                    2, 1, 0)
                aligned_wt = torch.bmm(
                    fc_layer0, sol.unsqueeze(0).repeat(fc_layer0.shape[0], 1, 1)
                ).permute(1, 2, 0)
                weight_pre_0 = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                fc_lay1 = weight_pre_1.view(sol.shape[0], num_cur_node, -1).permute(
                    2, 1, 0)
                layer1 = fc_lay1.permute(1, 2, 0)
                weight_pre_1 = layer1.contiguous().view(layer1.shape[0], -1)
                pass
        else:
            if position > 0:
                w0 = w0.T
                w1 = w1.T
            weight_pre_0 = torch.matmul(w0, sol)
            weight_pre_1 = w1
    else:
        if position > 0:
            w0 = w0.transpose(0, 1)
            w1 = w1.transpose(0, 1)
        weight_pre_0 = w0
        weight_pre_1 = w1
        t = sol.unsqueeze(0).repeat(weight_pre_0.shape[2], 1, 1)
        weight_pre_0 = torch.bmm(weight_pre_0.permute(2, 0, 1), t).permute(1, 2, 0)

    # ground_metric_object = GroundMetric(args)
    # layer_affinity = ground_metric_object.process(weight_pre_0.contiguous().view(weight_pre_0.shape[0], -1),
    #                                               weight_pre_1.contiguous().view(weight_pre_1.shape[0], -1))
    ground_metric = gm.Ground_Metric_GM_new(
        weight_pre_0, weight_pre_1, is_conv, is_bias,
        pre_conv, channel)
    layer_affinity = ground_metric.process_soft_affinity(p=4)
    print(f'weight_pre_0 size={weight_pre_0.shape}, weight_pre_1 size={weight_pre_1.shape}')
    return layer_affinity


def get_model_params(w0, sol, position=0):
    if len(w0.shape) == 2:
        if w0.shape[1] != sol.shape[0]:
            if position == 0:
                weight_pre_0 = w0
                fc_layer0 = weight_pre_0.view(weight_pre_0.shape[0], sol.shape[0], -1).permute(
                    2, 0, 1)
                aligned_wt = torch.bmm(
                    fc_layer0, sol.unsqueeze(0).repeat(fc_layer0.shape[0], 1, 1)
                ).permute(1, 2, 0)
                # weight_pre_0 = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                weight_pre_0 = aligned_wt.contiguous().view(weight_pre_0.shape[0], -1)
                pass
            else:
                num_cur_node = position
                weight_pre_0 = w0
                sol = sol.T
                fc_layer0 = weight_pre_0.view(sol.shape[0], num_cur_node, -1).permute(
                    2, 0, 1)
                aligned_wt = torch.bmm(
                    sol.unsqueeze(0).repeat(fc_layer0.shape[0], 1, 1), fc_layer0
                ).permute(1, 2, 0)
                # weight_pre_0 = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                weight_pre_0 = aligned_wt.contiguous().view(weight_pre_0.shape[0], -1)
                pass
        else:
            if position > 0:
                w0 = w0.T
            weight_pre_0 = torch.matmul(w0, sol)
            if position > 0:
                weight_pre_0 = weight_pre_0.T
    else:
        if position > 0:
            w0 = w0.transpose(0, 1)
        weight_pre_0 = w0
        t = sol.unsqueeze(0).repeat(weight_pre_0.shape[2], 1, 1)
        weight_pre_0 = torch.bmm(weight_pre_0.permute(2, 0, 1), t).permute(1, 2, 0)
        if position > 0:
            weight_pre_0 = weight_pre_0.transpose(0, 1)

    return weight_pre_0


def get_lap_affinity(args, w0, w1, sol, is_conv, is_bias, pre_conv, channel, position=0, is_ot=False):
    if len(w0.shape) == 2:
        if w0.shape[1] != sol.shape[0]:
            if position == 0:
                weight_pre_0 = w0
                fc_layer0 = weight_pre_0.view(weight_pre_0.shape[0], sol.shape[0], -1).permute(
                    2, 0, 1)
                aligned_wt = torch.bmm(
                    fc_layer0, sol.unsqueeze(0).repeat(fc_layer0.shape[0], 1, 1)
                ).permute(1, 2, 0)
                weight_pre_0 = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                weight_pre_1 = w1
            else:
                num_cur_node = position
                weight_pre_0 = w0
                weight_pre_1 = w1
                fc_layer0 = weight_pre_0.view(sol.shape[0], num_cur_node, -1).permute(
                    2, 1, 0)
                aligned_wt = torch.bmm(
                    fc_layer0, sol.unsqueeze(0).repeat(fc_layer0.shape[0], 1, 1)
                ).permute(1, 2, 0)
                weight_pre_0 = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                fc_lay1 = weight_pre_1.view(sol.shape[0], num_cur_node, -1).permute(
                    2, 1, 0)
                layer1 = fc_lay1.permute(1, 2, 0)
                weight_pre_1 = layer1.contiguous().view(layer1.shape[0], -1)
                pass
        else:
            if position > 0:
                w0 = w0.T
                w1 = w1.T
            weight_pre_0 = torch.matmul(w0, sol)
            weight_pre_1 = w1
    else:
        if position > 0:
            w0 = w0.transpose(0, 1)
            w1 = w1.transpose(0, 1)
        weight_pre_0 = w0
        weight_pre_1 = w1
        t = sol.unsqueeze(0).repeat(weight_pre_0.shape[2], 1, 1)
        weight_pre_0 = torch.bmm(weight_pre_0.permute(2, 0, 1), t).permute(1, 2, 0)

    if is_ot:
        ground_metric_object = GroundMetric(args)
        layer_affinity = ground_metric_object.process(weight_pre_0.contiguous().view(weight_pre_0.shape[0], -1),
                                                      weight_pre_1.contiguous().view(weight_pre_1.shape[0], -1))
    else:
        ground_metric = gm.Ground_Metric_GM_new(
            weight_pre_0, weight_pre_1, is_conv, is_bias,
            pre_conv, channel)
        layer_affinity = ground_metric.process_soft_affinity(p=4)
        layer_affinity = (layer_affinity / torch.mean(layer_affinity))
    return layer_affinity


def get_model(args, initial_solution_list, networks, pre_conv_out_channel, num_nodes_incremental,
              num_nodes_layers, cur_conv_list, conv_kernel_size_list, device, params_only=False):
    named_weight_list_0 = [named_parameter for named_parameter in networks[0].items()]
    aligned_wt_0 = [parameter.data for name, parameter in named_weight_list_0]
    idx = 0
    num_layers = len(aligned_wt_0)
    for num_before, num_cur, cur_conv, cur_kernel_size in \
            zip(num_nodes_incremental, num_nodes_layers, cur_conv_list, conv_kernel_size_list):
        # perm = solution[num_before:num_before + num_cur, num_before:num_before + num_cur]
        perm = initial_solution_list[idx + 1]
        if torch.sum(perm).item() != perm.shape[0]:
            perm_is_complete = False
        assert 'bias' not in named_weight_list_0[idx][0]
        if len(named_weight_list_0[idx][1].shape) == 4:
            aligned_wt_0[idx] = (
                    perm.transpose(0, 1).to(torch.float64) @ aligned_wt_0[idx].to(torch.float64).permute(2, 3, 0,
                                                                                                         1)) \
                .permute(2, 3, 0, 1)
        else:
            aligned_wt_0[idx] = perm.transpose(0, 1).to(torch.float64) @ aligned_wt_0[idx].to(torch.float64)
        idx += 1
        if idx >= num_layers:
            continue
        if 'bias' in named_weight_list_0[idx][0]:
            aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
            idx += 1
        if idx >= num_layers:
            continue
        if cur_conv and len(named_weight_list_0[idx][1].shape) == 2:
            aligned_wt_0[idx] = (aligned_wt_0[idx].to(torch.float64) \
                                 .reshape(aligned_wt_0[idx].shape[0], pre_conv_out_channel, -1) \
                                 .permute(0, 2, 1) \
                                 @ perm.to(torch.float64)) \
                .permute(0, 2, 1) \
                .reshape(aligned_wt_0[idx].shape[0], -1)
        elif len(named_weight_list_0[idx][1].shape) == 4:
            aligned_wt_0[idx] = ((aligned_wt_0[idx].to(torch.float64).permute(2, 3, 0, 1)) @ perm.to(torch.float64)) \
                .permute(2, 3, 0, 1)
        else:
            aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
    if params_only:
        return aligned_wt_0
    averaged_weights = []
    for idx, parameter in enumerate(networks[1].parameters()):
        averaged_weights.append((1 - args.ensemble_step) * aligned_wt_0[idx] + args.ensemble_step * parameter)
    fused_model = model.get_model_from_name(args)
    state_dict = fused_model.state_dict()
    for idx, (key, _) in enumerate(state_dict.items()):
        state_dict[key] = averaged_weights[idx]
    fused_model.load_state_dict(state_dict)
    return fused_model.to(device)


def graph_matching_align_gamf_parameters_multi(args, ws, w_rate, use_hungarian=False, is_ot=False):
    device = "cuda:{}".format(args.gpu_id) if args.gpu_id >= 0 else "cpu"
    solution_list, cur_conv_list, pre_conv_list, pre_conv_out_channel_list = \
        graph_matching_align_gamf_parameters(args, ws[0], ws[1], w_rate, use_hungarian=use_hungarian, perm_only=True)
    num_node_list = [len(e) for e in solution_list]
    num_iteration = 2000
    # ws[1] = ws[0]
    w_list = []
    for w in ws:
        w = {k: v.to(device) for k, v in w.items()}
        w_list.append(w)

    m_solution_list = []
    for i in range(len(ws)):
        solution_list = []
        for j in range(len(num_node_list)):
            if j == 0 or j == len(num_node_list) - 1:
                solution_list.append(
                    torch.eye(num_node_list[j], device=device))
            else:
                solution_list.append(
                    torch.rand([num_node_list[j], num_node_list[j]], device=device) * 0.1 - 0.05 + 1 / num_node_list[j])
        m_solution_list.append(solution_list)
    m_solution_list_before = []
    for i in range(len(ws)):
        solution_list = []
        for j in range(len(num_node_list)):
            if j == 0 or j == len(num_node_list) - 1:
                solution_list.append(
                    torch.eye(num_node_list[j], device=device))
            else:
                solution_list.append(
                    torch.rand([num_node_list[j], num_node_list[j]], device=device) * 0.1 - 0.05 + 1 / num_node_list[j])
        m_solution_list_before.append(solution_list)

    if is_ot:
        use_hungarian = True

    if use_hungarian:
        initial_solver = hungarian
        solver = hungarian
    else:
        initial_solver = Sinkhorn(max_iter=1, tau=10, epsilon=1.0e-10)
        solver = Sinkhorn(max_iter=200, tau=0.05, epsilon=1.0e-10)
    count = 0

    num_layers = len(w_list[0])
    w_list_detail = []
    for i in range(len(w_list)):
        model1 = w_list[i]
        n1 = total_node_num_parameters(model1)
        num_layers = len(w_list[i])
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
        named_weight_list_0 = [named_parameter for named_parameter in model1.items()]
        weight_list_0 = []
        input_num_nodes = 0
        output_num_nodes = 0
        # solution = torch.zeros([n1, n2], device=device)
        pre_conv_out_channel_list = []
        for idx, ((_, fc_layer0_weight), (_, fc_layer1_weight)) in \
                enumerate(zip(model1.items(), model1.items())):
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
            elif len(layer_shape) == 2:
                is_bias = False
                if not pre_bias:
                    pre_conv = is_conv
                    pre_conv_list.append(pre_conv)
                is_conv = False
                cur_conv_list.append(is_conv)
                fc_layer0_weight_data = fc_layer0_weight.data
            else:
                is_bias = True
                if not pre_bias:
                    pre_conv = is_conv
                    pre_conv_list.append(pre_conv)
                is_conv = False
                cur_conv_list.append(is_conv)
                fc_layer0_weight_data = fc_layer0_weight.data
            # if it's conv, update [pre_conv_out_channel]
            if is_conv:
                pre_conv_out_channel = num_nodes_cur

            weight_list_0.append(fc_layer0_weight_data)

            pre_conv_out_channel_list.append(pre_conv_out_channel)
            if is_bias is False:
                pre_conv_kernel_size = fc_layer0_weight.shape[3] if is_conv else None
                conv_kernel_size_list.append(pre_conv_kernel_size)
            if is_bias is False:
                num_nodes_before += num_nodes_pre
                num_nodes_incremental.append(num_nodes_before)
                num_nodes_layers.append(num_nodes_cur)

        w_list_detail.append(weight_list_0)

    for e in range(num_iteration):
        m_affinity_list = []
        for i in range(len(w_list)):
            solution_list = []
            for j in range(len(num_node_list)):
                solution_list.append(torch.zeros([num_node_list[j], num_node_list[j]], device=device))
            m_affinity_list.append(solution_list)

        for i in range(len(w_list)):
            for j in range(len(w_list)):
                if i == j:
                    continue

                for idx in range(num_layers - 1):
                    lap_prev = get_lap_affinity(args, w_list_detail[i][idx], w_list_detail[j][idx],
                                                m_solution_list[i][idx] @ m_solution_list[j][idx].T,
                                                cur_conv_list[idx], False, pre_conv_list[idx],
                                                int(w_list_detail[i][idx].shape[1] / pre_conv_out_channel_list[idx]), 0
                                                , is_ot)
                    lap_next = get_lap_affinity(args, w_list_detail[i][idx + 1], w_list_detail[j][idx + 1],
                                                m_solution_list[i][idx + 2] @ m_solution_list[j][idx + 2].T,
                                                cur_conv_list[idx + 1], False, pre_conv_list[idx + 1],
                                                int(w_list_detail[i][idx + 1].shape[1] / pre_conv_out_channel_list[
                                                    idx + 1]),
                                                len(m_solution_list[i][idx + 1]), is_ot)
                    if is_ot:
                        lap_affinity = lap_prev
                    else:
                        lap_affinity = lap_prev + lap_next
                    m_affinity_list[i][idx + 1] += lap_affinity @ m_solution_list[j][idx + 1]

        loss_list = []
        loss_list_before = []
        check_list = []
        for i in range(len(w_list)):
            check = 0
            loss = 0
            loss_before = 0
            for idx in range(num_layers - 1):
                affinity = m_affinity_list[i][idx + 1]
                # if not use_hungarian:
                #     rand_noise = torch.rand([num_node_list[idx + 1], num_node_list[idx + 1]],
                #                             device=device) * 0.0001 - 0.00005
                #     affinity_ = affinity + rand_noise
                # else:
                #     if count == 0:
                #         affinity_ = affinity + torch.eye(num_node_list[idx + 1], device=device) * 0.5 / num_node_list[
                #             idx + 1]
                #     else:
                #         affinity_ = affinity
                sol = solver(affinity)
                loss += ((sol - m_solution_list[i][idx + 1]) ** 2).sum().cpu().numpy()
                loss_before += ((sol - m_solution_list_before[i][idx + 1]) ** 2).sum().cpu().numpy()
                check += min(((sol - m_solution_list[i][idx + 1]) ** 2).sum().cpu().numpy(),
                             ((sol - m_solution_list_before[i][idx + 1]) ** 2).sum().cpu().numpy())
                m_solution_list_before[i][idx + 1] = m_solution_list[i][idx + 1]
                m_solution_list[i][idx + 1] = sol

            loss_list.append(loss)
            loss_list_before.append(loss_before)
            check_list.append(check)
        if use_hungarian:
            print('\nInteration: {}, Loss: {}, Loss_2: {}'.format(e, np.mean(loss_list), np.mean(loss_list_before)))
        else:
            print('\nInteration: {}, Loss: {}, Loss_2: {}, tau: {}, max: {}'.format(e, np.mean(loss_list),
                                                                                    np.mean(loss_list_before),
                                                                                    solver.tau,
                                                                                    solver.max_iter))
        print(check_list)
        for n in range(len(w_list)):
            max_list = []
            eye_list = []
            for i in range(num_layers + 1):
                x = m_solution_list[n][i].cpu().detach().numpy().flatten()
                z = heapq.nlargest(len(m_solution_list[n][i]), x)
                max_list.append(np.sum(z) / len(m_solution_list[n][i]))
                eye_list.append(int((m_solution_list[n][i] == torch.eye(len(m_solution_list[n][i]),
                                                                        device=device)).all().cpu().detach().numpy()))
            print(max_list)
            print(eye_list)

        # m_solution_list = []
        # solution_list = []
        # for j in range(len(num_node_list)):
        #     # solution_list.append(
        #     #     (torch.eye(num_node_list[j], device=device)))
        #     # solution_list.append(
        #     #     hungarian(torch.rand([num_node_list[j], num_node_list[j]], device=device)))
        #     if j == 0 or j == len(num_node_list) - 1:
        #         solution_list.append(
        #             torch.eye(num_node_list[j], device=device))
        #     else:
        #         # solution_list.append(
        #         #     hungarian(torch.rand([num_node_list[j], num_node_list[j]], device=device)))
        #         solution_list.append(
        #             torch.ones([num_node_list[j], num_node_list[j]], device=device) * 1 / num_node_list[j])
        # for i in range(len(ws)):
        #     m_solution_list.append(solution_list)

        # break
        check = np.mean(check_list)

        if not use_hungarian and (check < 2e-3 or count > 50):
        # if not use_hungarian and (check < 1e-2 or count > 50):
            if solver.tau > 0.005:
                solver.tau *= 0.9 # 0.95
                # solver.tau *= 0.8
                solver.max_iter += 100 # 200
                # solver.max_iter += 50
                count = 0
            else:
                # break
                use_hungarian = True
                solver = hungarian
                count = 0
        elif use_hungarian and (check < 2e-3 or count > 50):
            # elif use_hungarian and (check < 2e-3 or count > 50):
            break
        else:
            count += 1
        if e > num_iteration - 75:
            use_hungarian = True
            solver = hungarian
            count = 0


    params = [0 for _ in range(num_layers)]
    for i in range(len(w_list)):
        solution_list = [m_solution_list[i][k] for k in range(len(m_solution_list[i]))]
        assert (solution_list[0] == torch.eye(len(solution_list[0]), device=device)).all()
        assert (solution_list[-1] == torch.eye(len(solution_list[-1]), device=device)).all()
        for j in range(num_layers):
            layer_params_ = get_model_params(w_list_detail[i][j], solution_list[j], 0)
            layer_params = get_model_params(layer_params_, solution_list[j + 1], len(solution_list[j]))
            check_1 = w_list_detail[i][j].reshape(len(solution_list[j + 1]), len(solution_list[j]), -1)
            check_1 = solution_list[j + 1].T @ check_1.sum(axis=2) @ solution_list[j]
            check_2 = layer_params.reshape(len(solution_list[j + 1]), len(solution_list[j]), -1)
            check_2 = check_2.sum(axis=2)
            if ((check_2 - check_1) ** 2).sum() >= 1e-5:
                print(((check_2 - check_1) ** 2).sum())
            # assert ((check_2 - check_1) ** 2).sum() < 1e-5
            if len(layer_params.shape) == 3:
                conv_size = int(layer_params.shape[2] ** 0.5)
                layer_params = layer_params.reshape(layer_params.shape[0], layer_params.shape[1], conv_size,
                                                    conv_size)
            if i == 0:
                params[j] = layer_params * w_rate[i]
            else:
                params[j] += layer_params * w_rate[i]

    # named_weight_list_0 = [named_parameter for named_parameter in w_list[0].items()]
    # params = [parameter.data for name, parameter in named_weight_list_0]
    # for i in range(len(w_list)):
    #     if i == 0:
    #         continue
    #     solution_list = [m_solution_list[i][k] @ m_solution_list[0][k].T for k in range(len(m_solution_list[i]))]
    #     assert (solution_list[0] == torch.eye(len(solution_list[0]), device=device)).all()
    #     assert (solution_list[-1] == torch.eye(len(solution_list[-1]), device=device)).all()
    #     for j in range(num_layers):
    #         layer_params_ = get_model_params(w_list_detail[i][j], solution_list[j], 0)
    #         layer_params = get_model_params(layer_params_, solution_list[j + 1], len(solution_list[j]))
    #         check_1 = w_list_detail[i][j].reshape(len(solution_list[j + 1]), len(solution_list[j]), -1)
    #         check_1 = solution_list[j + 1].T @ check_1.sum(axis=2) @ solution_list[j]
    #         check_2 = layer_params.reshape(len(solution_list[j + 1]), len(solution_list[j]), -1)
    #         check_2 = check_2.sum(axis=2)
    #         assert torch.norm(check_2 - check_1) < 1e-5
    #         if len(layer_params.shape) == 3:
    #             conv_size = int(layer_params.shape[2] ** 0.5)
    #             layer_params = layer_params.reshape(layer_params.shape[0], layer_params.shape[1], conv_size,
    #                                                 conv_size)
    #         params[j] += layer_params

    # params = [p / len(w_list) for p in params]

    return params, True


def graph_matching_fusion_gamf(args, networks: list, test_loader, logger, use_hungarian=False):
    '''
    the function use graph matching technique to align each layer in networks[0] along with
        networks[1], and return a list that contains the averaged aligned parameters, following
        the original order of parameters in model.parameters()
    the averaging weights are specified in [args.ensemble_step, 1-args.ensemble_step]
    '''
    device = "cuda:{}".format(args.gpu_id) if args.gpu_id >= 0 else "cpu"
    n1 = total_node_num(network=networks[0])
    n2 = total_node_num(network=networks[1])
    assert (n1 == n2)
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
    weight_list_0 = []
    weight_list_1 = []
    input_num_nodes = 0
    output_num_nodes = 0
    solution = torch.zeros([n1, n2], device=device)
    pre_conv_out_channel_list = []
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

        weight_list_0.append(fc_layer0_weight_data)
        weight_list_1.append(fc_layer1_weight_data)

        if idx == 0:
            input_num_nodes = num_nodes_pre
            for a in range(num_nodes_pre):
                solution[a][a] = 1
        if idx == num_layers - 2 and 'bias' in named_weight_list_0[idx + 1][0] or \
                idx == num_layers - 1 and 'bias' not in named_weight_list_0[idx][0]:
            output_num_nodes = num_nodes_cur
            for a in range(num_nodes_cur):
                solution[num_nodes_before + num_nodes_pre + a][num_nodes_before + num_nodes_pre + a] = 1
        pre_conv_out_channel_list.append(pre_conv_out_channel)
        if is_bias is False:
            pre_conv_kernel_size = fc_layer0_weight.shape[3] if is_conv else None
            conv_kernel_size_list.append(pre_conv_kernel_size)
        if is_bias is False:
            num_nodes_before += num_nodes_pre
            num_nodes_incremental.append(num_nodes_before)
            num_nodes_layers.append(num_nodes_cur)

    # use_hungarian = True

    if use_hungarian:
        initial_solver = hungarian
        solver = hungarian
    else:
        initial_solver = Sinkhorn(max_iter=1, tau=10, epsilon=1.0e-10)
        solver = Sinkhorn(max_iter=20, tau=10, epsilon=1.0e-10)
    initial_solution_list = [solution[0: input_num_nodes, 0: input_num_nodes]]
    for i in range(num_layers - 1):
        num_nodes_cur = num_nodes_layers[i]
        # num_nodes_pre = num_nodes_layers[i - 1] if i > 0 else input_num_nodes
        initial_solution = torch.rand([num_nodes_cur, num_nodes_cur], device=device)
        # initial_solution = torch.ones([num_nodes_cur, num_nodes_cur], device=device) / num_nodes_cur
        initial_solution = initial_solution * 0.2 + 0.4
        initial_solution_ = initial_solver(initial_solution)
        initial_solution_list.append(initial_solution_)
    initial_solution_list.append(solution[-output_num_nodes:, -output_num_nodes:])
    num_epoch = 100
    # num_epoch = 30
    for e in range(num_epoch):
        loss = 0
        max_list = []
        std = 0
        # log_dict = {}
        # log_dict['test_losses'] = []
        # gm_model = get_model(args, initial_solution_list, networks, pre_conv_out_channel, num_nodes_incremental,
        #                         num_nodes_layers, cur_conv_list, conv_kernel_size_list, device)
        # routines.test(args, gm_model, test_loader, log_dict, logger)
        # random_list = np.random.permutation(num_layers - 1)
        random_list = range(num_layers - 1)
        for idx in range(num_layers - 1):
            i = random_list[idx]
            lap_prev = get_lap_affinity(args, weight_list_0[i], weight_list_1[i], initial_solution_list[i],
                                        cur_conv_list[i], False, pre_conv_list[i],
                                        int(weight_list_0[i].shape[1] / pre_conv_out_channel_list[i]), 0)
            lap_next = get_lap_affinity(args, weight_list_0[i + 1], weight_list_1[i + 1], initial_solution_list[i + 2],
                                        cur_conv_list[i + 1], False, pre_conv_list[i + 1],
                                        int(weight_list_0[i + 1].shape[1] / pre_conv_out_channel_list[i + 1]),
                                        len(initial_solution_list[i + 1]))
            # print(len(initial_solution_list[i]), len(initial_solution_list[i + 1]), len(initial_solution_list[i + 2]))
            # print(torch.mean(lap_prev).cpu().numpy(),
            #       torch.mean(lap_next).cpu().numpy())
            lap_affinity = lap_prev + lap_next
            solution = solver(lap_affinity)
            loss += ((solution - initial_solution_list[i + 1]) ** 2).sum().cpu().numpy()
            std += solution.cpu().detach().numpy().std()
            initial_solution_list[i + 1] = solution
            pass
        for i in range(num_layers + 1):
            x = initial_solution_list[i].cpu().detach().numpy().flatten()
            z = heapq.nlargest(len(initial_solution_list[i]), x)
            max_list.append(np.sum(z) / len(initial_solution_list[i]))
        if not use_hungarian:
            print(
                '\nEpoch: {}, Loss: {}, std: {} max iter: {}, tau: {}'.format(e, loss, std, solver.max_iter,
                                                                              solver.tau))
            print(max_list)
        else:
            print('\nEpoch: {}, Loss: {}, std: {}'.format(e, loss, std))
            print(max_list)
        if not use_hungarian:
            solver.tau *= 0.8
            if solver.tau <= 0.01:
                solver.tau = 0.01
            else:
                solver.max_iter += 100
            # if solver.tau < 0.1:
            #     solver = hungarian
            #     use_hungarian = True
        pass
    # if not use_hungarian:
    #     for i in range(num_layers - 1):
    #         initial_solution_list[i + 1] = hungarian(initial_solution_list[i + 1])
    aligned_wt_0 = [parameter.data for name, parameter in named_weight_list_0]
    idx = 0
    num_layers = len(aligned_wt_0)
    for num_before, num_cur, cur_conv, cur_kernel_size in \
            zip(num_nodes_incremental, num_nodes_layers, cur_conv_list, conv_kernel_size_list):
        # perm = solution[num_before:num_before + num_cur, num_before:num_before + num_cur]
        perm = initial_solution_list[idx + 1]
        if torch.sum(perm).item() != perm.shape[0]:
            perm_is_complete = False
        assert 'bias' not in named_weight_list_0[idx][0]
        if len(named_weight_list_0[idx][1].shape) == 4:
            aligned_wt_0[idx] = (
                    perm.transpose(0, 1).to(torch.float64) @ aligned_wt_0[idx].to(torch.float64).permute(2, 3, 0,
                                                                                                         1)) \
                .permute(2, 3, 0, 1)
        else:
            aligned_wt_0[idx] = perm.transpose(0, 1).to(torch.float64) @ aligned_wt_0[idx].to(torch.float64)
        idx += 1
        if idx >= num_layers:
            continue
        if 'bias' in named_weight_list_0[idx][0]:
            aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
            idx += 1
        if idx >= num_layers:
            continue
        if cur_conv and len(named_weight_list_0[idx][1].shape) == 2:
            aligned_wt_0[idx] = (aligned_wt_0[idx].to(torch.float64) \
                                 .reshape(aligned_wt_0[idx].shape[0], pre_conv_out_channel, -1) \
                                 .permute(0, 2, 1) \
                                 @ perm.to(torch.float64)) \
                .permute(0, 2, 1) \
                .reshape(aligned_wt_0[idx].shape[0], -1)
        elif len(named_weight_list_0[idx][1].shape) == 4:
            aligned_wt_0[idx] = ((aligned_wt_0[idx].to(torch.float64).permute(2, 3, 0, 1)) @ perm.to(torch.float64)) \
                .permute(2, 3, 0, 1)
        else:
            aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
    assert idx == num_layers

    averaged_weights = []
    for idx, parameter in enumerate(networks[1].parameters()):
        averaged_weights.append((1 - args.ensemble_step) * aligned_wt_0[idx] + args.ensemble_step * parameter)
    return averaged_weights, perm_is_complete


def graph_matching_fusion(args, networks: list):
    '''
    the function use graph matching technique to align each layer in networks[0] along with
        networks[1], and return a list that contains the averaged aligned parameters, following
        the original order of parameters in model.parameters()
    the averaging weights are specified in [args.ensemble_step, 1-args.ensemble_step]
    '''
    device = "cuda:{}".format(args.gpu_id) if args.gpu_id >= 0 else "cpu"
    n1 = total_node_num(network=networks[0])
    n2 = total_node_num(network=networks[1])
    assert (n1 == n2)
    affinity = torch.zeros([n1 * n2, n1 * n2], device=device)
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

        if is_bias is False:
            pre_conv_kernel_size = fc_layer0_weight.shape[3] if is_conv else None
            conv_kernel_size_list.append(pre_conv_kernel_size)
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
        if is_bias is False:
            num_nodes_before += num_nodes_pre
            num_nodes_incremental.append(num_nodes_before)
            num_nodes_layers.append(num_nodes_cur)

    solution = gb.gurobi_qap_solver(affinity, n1, n2, time_limit=300)

    aligned_wt_0 = [parameter.data for name, parameter in named_weight_list_0]
    idx = 0
    num_layers = len(aligned_wt_0)
    for num_before, num_cur, cur_conv, cur_kernel_size in \
            zip(num_nodes_incremental, num_nodes_layers, cur_conv_list, conv_kernel_size_list):
        perm = solution[num_before:num_before + num_cur, num_before:num_before + num_cur]
        if torch.sum(perm).item() != perm.shape[0]:
            perm_is_complete = False
        assert 'bias' not in named_weight_list_0[idx][0]
        if len(named_weight_list_0[idx][1].shape) == 4:
            aligned_wt_0[idx] = (perm.transpose(0, 1).to(torch.float64) @ \
                                 aligned_wt_0[idx].to(torch.float64).permute(2, 3, 0, 1)) \
                .permute(2, 3, 0, 1)
        else:
            aligned_wt_0[idx] = perm.transpose(0, 1).to(torch.float64) @ aligned_wt_0[idx].to(torch.float64)
        idx += 1
        if idx >= num_layers:
            continue
        if 'bias' in named_weight_list_0[idx][0]:
            aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
            idx += 1
        if idx >= num_layers:
            continue
        if cur_conv and len(named_weight_list_0[idx][1].shape) == 2:
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

    averaged_weights = []
    for idx, parameter in enumerate(networks[1].parameters()):
        averaged_weights.append((1 - args.ensemble_step) * aligned_wt_0[idx] + args.ensemble_step * parameter)
    return averaged_weights, perm_is_complete


def get_fused_model(args, networks: list, test_loader, logger, solver="Gurobi"):
    '''
    the input [parameters] is a list consisting of tensors
    '''
    device = "cuda:{}".format(args.gpu_id) if args.gpu_id >= 0 else "cpu"
    if solver == "Gurobi":
        parameters, perm_is_complete = graph_matching_fusion(args, networks)
    elif solver == "gamf":
        parameters, perm_is_complete = graph_matching_fusion_gamf(args, networks, test_loader, logger,
                                                                  use_hungarian=False)
    else:
        raise NotImplementedError
    fused_model = model.get_model_from_name(args)
    state_dict = fused_model.state_dict()
    for idx, (key, _) in enumerate(state_dict.items()):
        state_dict[key] = parameters[idx]
    fused_model.load_state_dict(state_dict)
    return fused_model.to(device), perm_is_complete


def gm_weight_align(args, w: torch.tensor, anchor: torch.tensor, device, w_rate, solver="Gurobi"):
    '''
    [w] and [anchor] are parameters for the ENTIRE network. 

    this function aligns the weight [w] with [anchor]
    returns the aligned [aligned_w]
    all the operations are done on cpu
    '''
    # args.gpu_id = args.gpu

    if solver == "Gurobi":
        parameters, perm_is_complete = graph_matching_align_parameters(args, w, anchor)
        aligned_w = {k: parameters[idx] for idx, (k, v) in enumerate(w.items())}
    elif solver == "gamf":
        parameters, perm_is_complete = graph_matching_align_gamf_parameters(args, w, anchor, w_rate,
                                                                            use_hungarian=False)
        aligned_w = {k: parameters[idx] for idx, (k, v) in enumerate(w.items())}
    elif solver == "gamf_multi":
        parameters, perm_is_complete = graph_matching_align_gamf_parameters_multi(args, w, w_rate, use_hungarian=False)
        aligned_w = {k: parameters[idx] for idx, (k, v) in enumerate(w[0].items())}
    elif solver == "ot_multi":
        parameters, perm_is_complete = graph_matching_align_gamf_parameters_multi(args, w, w_rate, use_hungarian=False, is_ot=True)
        aligned_w = {k: parameters[idx] for idx, (k, v) in enumerate(w[0].items())}
    elif solver == "ot":
        parameters, perm_is_complete = graph_matching_align_gamf_parameters(args, w, anchor, w_rate,
                                                                            use_hungarian=False, is_ot=True)
        aligned_w = {k: parameters[idx] for idx, (k, v) in enumerate(w.items())}
    else:
        print(solver)
        raise NotImplementedError

    return aligned_w
    pass


def graph_matching_align_parameters(args, weight: dict, anchor: dict):
    '''
    the function use graph matching technique to align each layer in networks[0] along with
        networks[1], and return a list that contains the averaged aligned parameters, following
        the original order of parameters in model.parameters()
    the averaging weights are specified in [args.ensemble_step, 1-args.ensemble_step]
    '''
    device = "cuda:{}".format(args.gpu_id) if args.gpu_id >= 0 else "cpu"
    weight = {k: v.to(device) for k, v in weight.items()}
    anchor = {k: v.to(device) for k, v in anchor.items()}
    n1 = total_node_num_parameters(weight)
    n2 = total_node_num_parameters(anchor)
    print(f'debug: n1={n1}, n2={n2}')
    assert (n1 == n2)
    affinity = torch.zeros([n1 * n2, n1 * n2], device=device)
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
    num_layers = len(weight)
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
    named_weight_list_0 = [(name, parameter) for name, parameter in weight.items()]
    for idx, (fc_layer0_weight, fc_layer1_weight) in \
            enumerate(zip(weight, anchor)):
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

        if is_bias is False:
            pre_conv_kernel_size = fc_layer0_weight.shape[3] if is_conv else None
            conv_kernel_size_list.append(pre_conv_kernel_size)
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
        if is_bias is False:
            num_nodes_before += num_nodes_pre
            num_nodes_incremental.append(num_nodes_before)
            num_nodes_layers.append(num_nodes_cur)

    solution = gb.gurobi_qap_solver(affinity, n1, n2, time_limit=300)

    aligned_wt_0 = [parameter.data for name, parameter in named_weight_list_0]
    idx = 0
    num_layers = len(aligned_wt_0)
    for num_before, num_cur, cur_conv, cur_kernel_size in \
            zip(num_nodes_incremental, num_nodes_layers, cur_conv_list, conv_kernel_size_list):
        perm = solution[num_before:num_before + num_cur, num_before:num_before + num_cur]
        if torch.sum(perm).item() != perm.shape[0]:
            perm_is_complete = False
        assert 'bias' not in named_weight_list_0[idx][0]
        if len(named_weight_list_0[idx][1].shape) == 4:
            aligned_wt_0[idx] = (perm.transpose(0, 1).to(torch.float64) @ \
                                 aligned_wt_0[idx].to(torch.float64).permute(2, 3, 0, 1)) \
                .permute(2, 3, 0, 1)
        else:
            aligned_wt_0[idx] = perm.transpose(0, 1).to(torch.float64) @ aligned_wt_0[idx].to(torch.float64)
        idx += 1
        if idx >= num_layers:
            continue
        if 'bias' in named_weight_list_0[idx][0]:
            aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
            idx += 1
        if idx >= num_layers:
            continue
        if cur_conv and len(named_weight_list_0[idx][1].shape) == 2:
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

    # averaged_weights = []
    # for idx, parameter in enumerate(anchor):
    #     averaged_weights.append((1 - args.ensemble_step) * aligned_wt_0[idx] + args.ensemble_step * parameter)
    aligned_wt_0 = {k: v.cpu() for k, v in aligned_wt_0.items()}
    return aligned_wt_0, perm_is_complete


def graph_matching_align_gamf_parameters(args, w: dict, anchor: dict, w_rate, use_hungarian=False, perm_only=False,
                                         is_ot=False):
    '''
    the function use graph matching technique to align each layer in networks[0] along with
        networks[1], and return a list that contains the averaged aligned parameters, following
        the original order of parameters in model.parameters()
    the averaging weights are specified in [args.ensemble_step, 1-args.ensemble_step]
    '''
    device = "cuda:{}".format(args.gpu_id) if args.gpu_id >= 0 else "cpu"
    w = {k: v.to(device) for k, v in w.items()}
    anchor = {k: v.to(device) for k, v in anchor.items()}

    # debug
    aha1 = {name: parameter.shape for name, parameter in w.items()}
    aha2 = {name: parameter.shape for name, parameter in anchor.items()}
    print(f'w is \n{aha1}, \nand anchor is \n{aha2}')
    # end
    n1 = total_node_num_parameters(w)
    n2 = total_node_num_parameters(anchor)
    print(f'debug: n1={n1}, n2={n2}')
    assert (n1 == n2)
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
    num_layers = len(w)
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
    # named_weight_list_0 = [named_parameter for named_parameter in networks[0].named_parameters()]
    named_weight_list_0 = [(name, parameter) for name, parameter in w.items()]
    weight_list_0 = []
    weight_list_1 = []
    input_num_nodes = 0
    output_num_nodes = 0
    solution = torch.zeros([n1, n2], device=device)
    pre_conv_out_channel_list = []
    # for idx, ((_, fc_layer0_weight), (_, fc_layer1_weight)) in \
    #         enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):
    for idx, ((_, fc_layer0_weight), (_, fc_layer1_weight)) in \
            enumerate(zip(w.items(), anchor.items())):
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

        weight_list_0.append(fc_layer0_weight_data)
        weight_list_1.append(fc_layer1_weight_data)

        if idx == 0:
            input_num_nodes = num_nodes_pre
            for a in range(num_nodes_pre):
                solution[a][a] = 1
        if idx == num_layers - 2 and 'bias' in named_weight_list_0[idx + 1][0] or \
                idx == num_layers - 1 and 'bias' not in named_weight_list_0[idx][0]:
            output_num_nodes = num_nodes_cur
            for a in range(num_nodes_cur):
                solution[num_nodes_before + num_nodes_pre + a][num_nodes_before + num_nodes_pre + a] = 1
        pre_conv_out_channel_list.append(pre_conv_out_channel)
        if is_bias is False:
            pre_conv_kernel_size = fc_layer0_weight.shape[3] if is_conv else None
            conv_kernel_size_list.append(pre_conv_kernel_size)
        if is_bias is False:
            num_nodes_before += num_nodes_pre
            num_nodes_incremental.append(num_nodes_before)
            num_nodes_layers.append(num_nodes_cur)

    if is_ot:
        use_hungarian = True

    if use_hungarian:
        initial_solver = hungarian
        solver = hungarian
    else:
        initial_solver = Sinkhorn(max_iter=1, tau=10, epsilon=1.0e-10)
        solver = Sinkhorn(max_iter=20, tau=10, epsilon=1.0e-10)
    initial_solution_list = [solution[0: input_num_nodes, 0: input_num_nodes]]
    for i in range(num_layers - 1):
        num_nodes_cur = num_nodes_layers[i]
        # num_nodes_pre = num_nodes_layers[i - 1] if i > 0 else input_num_nodes
        initial_solution = torch.rand([num_nodes_cur, num_nodes_cur], device=device)
        # initial_solution = torch.ones([num_nodes_cur, num_nodes_cur], device=device) / num_nodes_cur
        initial_solution = initial_solution * 0.2 + 0.4
        initial_solution_ = initial_solver(initial_solution)
        initial_solution_list.append(initial_solution_)
    initial_solution_list.append(solution[-output_num_nodes:, -output_num_nodes:])
    # num_epoch = 100

    if perm_only:
        return initial_solution_list, cur_conv_list, pre_conv_list, pre_conv_out_channel_list

    num_epoch = 50
    for e in range(num_epoch):
        loss = 0
        max_list = []
        std = 0
        log_dict = {}
        log_dict['test_losses'] = []
        # gm_model = get_model(args, initial_solution_list, networks, pre_conv_out_channel, num_nodes_incremental,
        #                         num_nodes_layers, cur_conv_list, conv_kernel_size_list, device)
        # gm_model = get_model_parameters(args, initial_solution_list, w, anchor, pre_conv_out_channel, num_nodes_incremental,
        # num_nodes_layers, cur_conv_list, conv_kernel_size_list, device)
        # routines.test(args, gm_model, test_loader, log_dict, logger)
        # random_list = np.random.permutation(num_layers - 1)
        random_list = range(num_layers - 1)
        for idx in range(num_layers - 1):
            i = random_list[idx]
            lap_prev = get_lap_affinity(args, weight_list_0[i], weight_list_1[i], initial_solution_list[i],
                                        cur_conv_list[i], False, pre_conv_list[i],
                                        int(weight_list_0[i].shape[1] / pre_conv_out_channel_list[i]), 0, is_ot)
            lap_next = get_lap_affinity(args, weight_list_0[i + 1], weight_list_1[i + 1], initial_solution_list[i + 2],
                                        cur_conv_list[i + 1], False, pre_conv_list[i + 1],
                                        int(weight_list_0[i + 1].shape[1] / pre_conv_out_channel_list[i + 1]),
                                        len(initial_solution_list[i + 1]), is_ot)
            # print(len(initial_solution_list[i]), len(initial_solution_list[i + 1]), len(initial_solution_list[i + 2]))
            # print(torch.mean(lap_prev).cpu().numpy(),
            #       torch.mean(lap_next).cpu().numpy())
            if is_ot:
                lap_affinity = lap_prev
            else:
                lap_affinity = lap_prev + lap_next
            solution = solver(lap_affinity)
            loss += ((solution - initial_solution_list[i + 1]) ** 2).sum().cpu().numpy()
            std += solution.cpu().detach().numpy().std()
            initial_solution_list[i + 1] = solution
            pass
        if is_ot:
            break
        for i in range(num_layers + 1):
            x = initial_solution_list[i].cpu().detach().numpy().flatten()
            z = heapq.nlargest(len(initial_solution_list[i]), x)
            max_list.append(np.sum(z) / len(initial_solution_list[i]))

        if not use_hungarian:
            print(
                '\nEpoch: {}, Loss: {}, std: {} max iter: {}, tau: {}'.format(e, loss, std, solver.max_iter,
                                                                              solver.tau))
            print(max_list)
        else:
            print('\nEpoch: {}, Loss: {}, std: {}'.format(e, loss, std))
            print(max_list)
        if not use_hungarian:
            solver.tau *= 0.8
            # solver.tau *= 0.9
            if solver.tau <= 0.01:
                # if solver.tau <= 0.05:
                solver.tau = 0.01
                # solver.tau = 0.05
            else:
                solver.max_iter += 50
                # solver.max_iter += 50
            # if solver.tau < 0.1:
            #     solver = hungarian
            #     use_hungarian = True
        pass
    # if not use_hungarian:
    #     for i in range(num_layers - 1):
    #         initial_solution_list[i + 1] = hungarian(initial_solution_list[i + 1])
    aligned_wt_0 = [parameter.data for name, parameter in named_weight_list_0]
    idx = 0
    num_layers = len(aligned_wt_0)
    for num_before, num_cur, cur_conv, cur_kernel_size in \
            zip(num_nodes_incremental, num_nodes_layers, cur_conv_list, conv_kernel_size_list):
        # perm = solution[num_before:num_before + num_cur, num_before:num_before + num_cur]
        perm = initial_solution_list[idx + 1]
        if torch.sum(perm).item() != perm.shape[0]:
            perm_is_complete = False
        assert 'bias' not in named_weight_list_0[idx][0]
        if len(named_weight_list_0[idx][1].shape) == 4:
            aligned_wt_0[idx] = (
                    perm.transpose(0, 1).to(torch.float64) @ aligned_wt_0[idx].to(torch.float64).permute(2, 3, 0,
                                                                                                         1)) \
                .permute(2, 3, 0, 1)
        else:
            aligned_wt_0[idx] = perm.transpose(0, 1).to(torch.float64) @ aligned_wt_0[idx].to(torch.float64)
        idx += 1
        if idx >= num_layers:
            continue
        if 'bias' in named_weight_list_0[idx][0]:
            aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
            idx += 1
        if idx >= num_layers:
            continue
        if cur_conv and len(named_weight_list_0[idx][1].shape) == 2:
            aligned_wt_0[idx] = (aligned_wt_0[idx].to(torch.float64) \
                                 .reshape(aligned_wt_0[idx].shape[0], pre_conv_out_channel, -1) \
                                 .permute(0, 2, 1) \
                                 @ perm.to(torch.float64)) \
                .permute(0, 2, 1) \
                .reshape(aligned_wt_0[idx].shape[0], -1)
        elif len(named_weight_list_0[idx][1].shape) == 4:
            aligned_wt_0[idx] = ((aligned_wt_0[idx].to(torch.float64).permute(2, 3, 0, 1)) @ perm.to(torch.float64)) \
                .permute(2, 3, 0, 1)
        else:
            aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
    assert idx == num_layers

    # averaged_weights = []
    # for idx, parameter in enumerate(networks[1].parameters()):
    #     averaged_weights.append((1 - args.ensemble_step) * aligned_wt_0[idx] + args.ensemble_step * parameter)
    aligned_wt_0 = [v.cpu() for v in aligned_wt_0]
    return aligned_wt_0, perm_is_complete


def graph_matching_fusion_gamf_new(args, networks: list, test_loader, logger, use_hungarian=False, perm_only=False):
    '''
    the function use graph matching technique to align each layer in networks[0] along with
        networks[1], and return a list that contains the averaged aligned parameters, following
        the original order of parameters in model.parameters()
    the averaging weights are specified in [args.ensemble_step, 1-args.ensemble_step]
    '''
    device = "cuda:{}".format(args.gpu_id) if args.gpu_id >= 0 else "cpu"
    n1 = total_node_num(network=networks[0])
    n2 = total_node_num(network=networks[1])
    assert (n1 == n2)
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
    weight_list_0 = []
    weight_list_1 = []
    input_num_nodes = 0
    output_num_nodes = 0
    solution = torch.zeros([n1, n2], device=device)
    pre_conv_out_channel_list = []
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

        weight_list_0.append(fc_layer0_weight_data)
        weight_list_1.append(fc_layer1_weight_data)

        if idx == 0:
            input_num_nodes = num_nodes_pre
            for a in range(num_nodes_pre):
                solution[a][a] = 1
        if idx == num_layers - 2 and 'bias' in named_weight_list_0[idx + 1][0] or \
                idx == num_layers - 1 and 'bias' not in named_weight_list_0[idx][0]:
            output_num_nodes = num_nodes_cur
            for a in range(num_nodes_cur):
                solution[num_nodes_before + num_nodes_pre + a][num_nodes_before + num_nodes_pre + a] = 1
        pre_conv_out_channel_list.append(pre_conv_out_channel)
        if is_bias is False:
            pre_conv_kernel_size = fc_layer0_weight.shape[3] if is_conv else None
            conv_kernel_size_list.append(pre_conv_kernel_size)
        if is_bias is False:
            num_nodes_before += num_nodes_pre
            num_nodes_incremental.append(num_nodes_before)
            num_nodes_layers.append(num_nodes_cur)

    # use_hungarian = True

    if use_hungarian:
        initial_solver = hungarian
        solver = hungarian
    else:
        initial_solver = Sinkhorn(max_iter=1, tau=10, epsilon=1.0e-10)
        solver = Sinkhorn(max_iter=20, tau=10, epsilon=1.0e-10)
    initial_solution_list = [solution[0: input_num_nodes, 0: input_num_nodes]]
    for i in range(num_layers - 1):
        num_nodes_cur = num_nodes_layers[i]
        # num_nodes_pre = num_nodes_layers[i - 1] if i > 0 else input_num_nodes
        initial_solution = torch.rand([num_nodes_cur, num_nodes_cur], device=device)
        # initial_solution = torch.ones([num_nodes_cur, num_nodes_cur], device=device) / num_nodes_cur
        # initial_solution = initial_solution * 0.2 + 0.4
        initial_solution_ = initial_solver(initial_solution)
        initial_solution_list.append(initial_solution_)
    initial_solution_list.append(solution[-output_num_nodes:, -output_num_nodes:])

    if perm_only:
        return initial_solution_list, cur_conv_list, pre_conv_list, pre_conv_out_channel_list

    num_epoch = 75
    iter_boost = 50
    for e in range(num_epoch):
        loss = 0
        max_list = []
        std = 0
        log_dict = {}
        log_dict['test_losses'] = []
        # gm_model = get_model(args, initial_solution_list, networks, pre_conv_out_channel, num_nodes_incremental,
        #                      num_nodes_layers, cur_conv_list, conv_kernel_size_list, device)
        # routines.test(args, gm_model, test_loader, log_dict, logger)
        # random_list = np.random.permutation(num_layers - 1)
        random_list = range(num_layers - 1)
        for idx in range(num_layers - 1):
            i = random_list[idx]
            lap_prev = get_lap_affinity_new(args, weight_list_0[i], weight_list_1[i], initial_solution_list[i],
                                            cur_conv_list[i], False, pre_conv_list[i],
                                            int(weight_list_0[i].shape[1] / pre_conv_out_channel_list[i]), 0)
            lap_next = get_lap_affinity_new(args, weight_list_0[i + 1], weight_list_1[i + 1],
                                            initial_solution_list[i + 2],
                                            cur_conv_list[i + 1], False, pre_conv_list[i + 1],
                                            int(weight_list_0[i + 1].shape[1] / pre_conv_out_channel_list[i + 1]),
                                            len(initial_solution_list[i + 1]))
            # print(len(initial_solution_list[i]), len(initial_solution_list[i + 1]), len(initial_solution_list[i + 2]))
            # print(torch.mean(lap_prev).cpu().numpy(),
            #       torch.mean(lap_next).cpu().numpy())

            '''
            debug
            '''
            print(f'initial_solution_list[i] size = \n{initial_solution_list[i].shape} \
                \ninitial_solution_list[i+1] size = \n{initial_solution_list[i + 1].shape} \
                \ninitial_solution_list[i+2] size = \n{initial_solution_list[i + 2].shape} \
                \nlap_prev size = \n{lap_prev.shape}\nlap_next size = \n{lap_next.shape} \
                \nweight_list_0[i] size = \n{weight_list_0[i].shape} \
                \nweight_list_0[i+1] size = \n{weight_list_0[i + 1].shape}')
            '''
            end
            '''

            lap_affinity = lap_prev + lap_next
            solution = solver(lap_affinity)
            loss += ((solution - initial_solution_list[i + 1]) ** 2).sum().cpu().numpy()
            std += solution.cpu().detach().numpy().std()
            initial_solution_list[i + 1] = solution
            pass
        for i in range(num_layers + 1):
            x = initial_solution_list[i].cpu().detach().numpy().flatten()
            z = heapq.nlargest(len(initial_solution_list[i]), x)
            max_list.append(np.sum(z) / len(initial_solution_list[i]))
        if not use_hungarian:
            print(
                '\nEpoch: {}, Loss: {}, std: {} max iter: {}, tau: {}'.format(e, loss, std, solver.max_iter,
                                                                              solver.tau))
            print(max_list)
        else:
            print('\nEpoch: {}, Loss: {}, std: {}'.format(e, loss, std))
            print(max_list)
        if not use_hungarian:
            solver.tau *= 0.9
            if solver.tau <= 0.05:
                solver.tau = 0.05
            else:
                solver.max_iter += iter_boost
                # if solver.tau < 0.1:
                #     iter_boost = 500
            # if solver.tau < 0.1:
            #     solver = hungarian
            #     use_hungarian = True
        pass
    # if not use_hungarian:
    #     for i in range(num_layers - 1):
    #         initial_solution_list[i + 1] = hungarian(initial_solution_list[i + 1])

    aligned_wt_0 = [parameter.data for name, parameter in named_weight_list_0]
    idx = 0
    num_layers = len(aligned_wt_0)
    for num_before, num_cur, cur_conv, cur_kernel_size in \
            zip(num_nodes_incremental, num_nodes_layers, cur_conv_list, conv_kernel_size_list):
        # perm = solution[num_before:num_before + num_cur, num_before:num_before + num_cur]
        perm = initial_solution_list[idx + 1]
        if torch.sum(perm).item() != perm.shape[0]:
            perm_is_complete = False
        assert 'bias' not in named_weight_list_0[idx][0]
        if len(named_weight_list_0[idx][1].shape) == 4:
            aligned_wt_0[idx] = (
                    perm.transpose(0, 1).to(torch.float64) @ aligned_wt_0[idx].to(torch.float64).permute(2, 3, 0,
                                                                                                         1)) \
                .permute(2, 3, 0, 1)
        else:
            aligned_wt_0[idx] = perm.transpose(0, 1).to(torch.float64) @ aligned_wt_0[idx].to(torch.float64)
        idx += 1
        if idx >= num_layers:
            continue
        if 'bias' in named_weight_list_0[idx][0]:
            aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
            idx += 1
        if idx >= num_layers:
            continue
        if cur_conv and len(named_weight_list_0[idx][1].shape) == 2:
            aligned_wt_0[idx] = (aligned_wt_0[idx].to(torch.float64) \
                                 .reshape(aligned_wt_0[idx].shape[0], pre_conv_out_channel, -1) \
                                 .permute(0, 2, 1) \
                                 @ perm.to(torch.float64)) \
                .permute(0, 2, 1) \
                .reshape(aligned_wt_0[idx].shape[0], -1)
        elif len(named_weight_list_0[idx][1].shape) == 4:
            aligned_wt_0[idx] = ((aligned_wt_0[idx].to(torch.float64).permute(2, 3, 0, 1)) @ perm.to(torch.float64)) \
                .permute(2, 3, 0, 1)
        else:
            aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
    assert idx == num_layers

    averaged_weights = []
    for idx, parameter in enumerate(networks[1].parameters()):
        averaged_weights.append((1 - args.ensemble_step) * aligned_wt_0[idx] + args.ensemble_step * parameter)
    return averaged_weights, perm_is_complete


# def get_model_parameters(args, initial_solution_list, w:dict, anchor:dict, pre_conv_out_channel, num_nodes_incremental,
#               num_nodes_layers, cur_conv_list, conv_kernel_size_list, device):
#     perm_is_complete = True
#     # named_weight_list_0 = [named_parameter for named_parameter in networks[0].named_parameters()]
#     named_weight_list_0 = [(name, parameter) for name, parameter in w.items()]
#     aligned_wt_0 = [parameter.data for name, parameter in named_weight_list_0]
#     idx = 0
#     num_layers = len(aligned_wt_0)
#     for num_before, num_cur, cur_conv, cur_kernel_size in \
#             zip(num_nodes_incremental, num_nodes_layers, cur_conv_list, conv_kernel_size_list):
#         # perm = solution[num_before:num_before + num_cur, num_before:num_before + num_cur]
#         perm = initial_solution_list[idx + 1]
#         if torch.sum(perm).item() != perm.shape[0]:
#             perm_is_complete = False
#         assert 'bias' not in named_weight_list_0[idx][0]
#         if len(named_weight_list_0[idx][1].shape) == 4:
#             aligned_wt_0[idx] = (
#                     perm.transpose(0, 1).to(torch.float64) @ aligned_wt_0[idx].to(torch.float64).permute(2, 3, 0,
#                                                                                                          1)) \
#                 .permute(2, 3, 0, 1)
#         else:
#             aligned_wt_0[idx] = perm.transpose(0, 1).to(torch.float64) @ aligned_wt_0[idx].to(torch.float64)
#         idx += 1
#         if idx >= num_layers:
#             continue
#         if 'bias' in named_weight_list_0[idx][0]:
#             aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
#             idx += 1
#         if idx >= num_layers:
#             continue
#         if cur_conv and len(named_weight_list_0[idx][1].shape) == 2:
#             aligned_wt_0[idx] = (aligned_wt_0[idx].to(torch.float64) \
#                                  .reshape(aligned_wt_0[idx].shape[0], pre_conv_out_channel, -1) \
#                                  .permute(0, 2, 1) \
#                                  @ perm.to(torch.float64)) \
#                 .permute(0, 2, 1) \
#                 .reshape(aligned_wt_0[idx].shape[0], -1)
#         elif len(named_weight_list_0[idx][1].shape) == 4:
#             aligned_wt_0[idx] = ((aligned_wt_0[idx].to(torch.float64).permute(2, 3, 0, 1)) @ perm.to(torch.float64)) \
#                 .permute(2, 3, 0, 1)
#         else:
#             aligned_wt_0[idx] = aligned_wt_0[idx].to(torch.float64) @ perm.to(torch.float64)
#     averaged_weights = []
#     for idx, parameter in enumerate(networks[1].parameters()):
#         averaged_weights.append((1 - args.ensemble_step) * aligned_wt_0[idx] + args.ensemble_step * parameter)
#     fused_model = model.get_model_from_name(args)
#     state_dict = fused_model.state_dict()
#     for idx, (key, _) in enumerate(state_dict.items()):
#         state_dict[key] = averaged_weights[idx]
#     fused_model.load_state_dict(state_dict)
#     return fused_model.to(device)

if __name__ == '__main__':
    import model_gm
    from utils_gm import Namespace

    args = Namespace(
        model='lenet',
        dataset='cifar10',
        data_dir='./FedML/data/cifar10',
        partition_method='hetero',
        partition_alpha='0.5',
        batch_size=256,
        client_optimizer='adam',
        lr=0.001,
        wd=0.001,
        epochs=1,
        client_num_in_total=10,
        client_num_per_round=10,
        comm_round=30,
        frequency_of_the_test=30,
        gpu=0,
        gpu_id=-1,
        ci=0,
        fusion_mode='fusion_gamf'
    )
    model1 = model_gm.get_model_from_name(name='lenet')
    model2 = model_gm.get_model_from_name(name='lenet')
    w = {k: v for k, v in model1.named_parameters()}
    anchor = {k: v for k, v in model2.named_parameters()}
    # a = graph_matching_align_gamf_parameters( args, w, anchor )
    b = graph_matching_fusion_gamf_new(args, [model1, model2], None, None)
    print(b)
