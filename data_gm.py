import torch
import torchvision
import sys
PATH_TO_CIFAR = "./cifar_gm/"
sys.path.append(PATH_TO_CIFAR)
import train as cifar_train

def get_inp_tar(dataset):
    return dataset.data.view(dataset.data.shape[0], -1).float(), dataset.targets

def get_mnist_dataset(root, is_train, to_download, return_tensor=False):
    mnist = torchvision.datasets.MNIST(root, train=is_train, download=to_download,
                    transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                       # only 1 channel
                       (0.1307,), (0.3081,))
                 ]))

    if not return_tensor:
        return mnist
    else:
        return get_inp_tar(mnist)


def get_dataloader(args, unit_batch = False, no_randomness=False):
    if unit_batch:
        bsz = (1, 1)
    else:
        bsz = (args.batch_size_train, args.batch_size_test)

    if no_randomness:
        enable_shuffle = False
    else:
        enable_shuffle = True

    '''
    lower() method returns the lower-case version of the original string
    '''
    if args.dataset.lower() == 'mnist':

        train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('./files/', train=True, download=args.to_download,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           # only 1 channel
                                           (0.1307,), (0.3081,))
                                     ])),
          batch_size=bsz[0], shuffle=enable_shuffle
        )


        test_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('./files/', train=False, download=args.to_download,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                     ])),
          batch_size=bsz[1], shuffle=enable_shuffle
        )

        return train_loader, test_loader

    elif args.dataset.lower() == 'cifar10':
        if args.cifar_style_data:
            train_loader, test_loader = cifar_train.get_dataset(args.config)
        else:

            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.CIFAR10('./data/', train=True, download=args.to_download,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   # Note this normalization is not same as in MNIST
                                                   # (mean_ch1, mean_ch2, mean_ch3), (std1, std2, std3)
                                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ])),
                batch_size=bsz[0], shuffle=enable_shuffle
            )

            test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.CIFAR10('./data/', train=False, download=args.to_download,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   # (mean_ch1, mean_ch2, mean_ch3), (std1, std2, std3)
                                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ])),
                batch_size=bsz[1], shuffle=enable_shuffle
            )

        return train_loader, test_loader


def get_federated_data_loader( args, weights=[0.5, 0.5], no_randomness=False, non_iid=True ):
    bsz = (args.batch_size_train, args.batch_size_test)
    if no_randomness:
        enable_shuffle = False
    else:
        enable_shuffle = True
    '''
    lower() method returns the lower-case version of the original string
    '''
    if args.dataset.lower() == 'mnist':
        if non_iid is False:
            '''
            iid version
            '''
            train_set = torchvision.datasets.MNIST('./files/', train=True, download=args.to_download,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                # only 1 channel
                                                (0.1307,), (0.3081,))]))

            test_set = torchvision.datasets.MNIST('./files/', train=False, download=args.to_download,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))]))
        else:
            '''
            TODO: non-iid version
            '''
            return None

    elif args.dataset.lower() == 'cifar10':
        if non_iid is False:
            '''
            iid version
            '''
            if args.cifar_style_data:
                # TODO: implement the funcion [get_federated_dataset]
                return cifar_train.get_federated_dataset(args.config)
            else:
                train_set = torchvision.datasets.CIFAR10('./data/', train=True, download=args.to_download,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                # Note this normalization is not same as in MNIST
                                                # (mean_ch1, mean_ch2, mean_ch3), (std1, std2, std3)
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

                test_set = torchvision.datasets.CIFAR10('./data/', train=False, download=args.to_download,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                # (mean_ch1, mean_ch2, mean_ch3), (std1, std2, std3)
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        else:
            '''
            TODO: non-iid version
            '''
            return None
    '''
    split the training set and return the training loaders
    '''
    train_set_1, train_set_2 = torch.utils.data.random_split(
                                train_set, 
                                [   int( weights[0] * len( train_set ) ), 
                                    len( train_set ) - int( weights[0] * len( train_set ) )])
    train_loader_1 = torch.utils.data.DataLoader( train_set_1, batch_size=bsz[0], shuffle=enable_shuffle )
    train_loader_2 = torch.utils.data.DataLoader( train_set_2, batch_size=bsz[0], shuffle=enable_shuffle )
    '''
    directly return the testing loaders
    '''
    test_loader = torch.utils.data.DataLoader( test_set, batch_size=bsz[1], shuffle=enable_shuffle )

    return train_loader_1, train_loader_2, test_loader