import torch
from model_gm import get_model_from_name
import torch.nn.functional as F

def get_pretrained_model(args, path, data_separated=False, idx=-1):
    model = get_model_from_name(args, idx=idx)

    if args.gpu_id != -1:
        state = torch.load(
            path,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cuda:' + str(args.gpu_id))
            ),
        )
    else:
        state = torch.load(
            path,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )


    model_state_dict = state['model_state_dict']

    if 'test_accuracy' not in state:
        state['test_accuracy'] = -1

    if 'epoch' not in state:
        state['epoch'] = -1

    if not data_separated:
        print("Loading model at path {} which had accuracy {} and at epoch {}".format(path, state['test_accuracy'],
                                                                                  state['epoch']))
    else:
        print("Loading model at path {} which had local accuracy {} and overall accuracy {} for choice {} at epoch {}".format(path,
            state['local_test_accuracy'], state['test_accuracy'], state['choice'], state['epoch']))

    model.load_state_dict(model_state_dict)

    if args.gpu_id != -1:
        model = model.cuda(args.gpu_id)

    if not data_separated:
        return model, state['test_accuracy']
    else:
        return model, state['test_accuracy'], state['local_test_accuracy']


def test(args, network, test_loader, log_dict, debug=False, return_loss=False, is_local=False):
    '''
    test the accuracies and loss of the model specified in [network] on the dataset specified
        by [test_loader]
    '''
    # turn the mode from train to eval, so that layers like "dropout" would be turned off
    network.eval()  
    test_loss = 0
    correct = 0
    if is_local:
        print("\nTesting in local mode")
    else:
        print("\nTesting in global mode")

    if args.dataset.lower() == 'cifar10':
        cifar_criterion = torch.nn.CrossEntropyLoss()

    #   with torch.no_grad():
    for data, target in test_loader:
        # print(data.shape, target.shape)
        # if len(target.shape)==1:
        #     data = data.unsqueeze(0)
        #     target = target.unsqueeze(0)
        # print(data, target)
        if args.gpu_id!=-1:
            data = data.cuda(args.gpu_id)
            target = target.cuda(args.gpu_id)

        output = network(data)
        if debug:
            print("output is ", output)

        if args.dataset.lower() == 'cifar10':
            # mnist models return log_softmax outputs, while cifar ones return raw values!
            test_loss += cifar_criterion(output, target).item()
        elif args.dataset.lower() == 'mnist':
            test_loss += F.nll_loss(output, target, size_average=False).item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    # Alexanderia
    # print("size of test_loader dataset: ", len(test_loader.dataset))
    test_loss /= len(test_loader.dataset)
    if is_local:
        string_info = 'local_test'
    else:
        string_info = 'test'
    log_dict['{}_losses'.format(string_info)].append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    ans = (float(correct) * 100.0) / len(test_loader.dataset)

    if not return_loss:
        return ans
    else:
        return ans, test_loss