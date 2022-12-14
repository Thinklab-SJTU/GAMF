import sys
PATH_TO_CIFAR = "./cifar_gm/"
sys.path.append(PATH_TO_CIFAR)
# import train as cifar_train

class Namespace():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def _get_config(args):
    print('refactored get_config')
    import hyperparameters.vgg11_cifar10_baseline as cifar10_vgg_hyperparams  # previously vgg_hyperparams
    import hyperparameters.vgg11_half_cifar10_baseline as cifar10_vgg_hyperparams_half
    import hyperparameters.vgg11_doub_cifar10_baseline as cifar10_vgg_hyperparams_doub
    import hyperparameters.vgg11_quad_cifar10_baseline as cifar10_vgg_hyperparams_quad
    import hyperparameters.resnet18_nobias_cifar10_baseline as cifar10_resnet18_nobias_hyperparams
    import hyperparameters.resnet18_nobias_nobn_cifar10_baseline as cifar10_resnet18_nobias_nobn_hyperparams
    import hyperparameters.mlpnet_cifar10_baseline as mlpnet_hyperparams

    config = None
    second_config = None

    if args.dataset.lower() == 'cifar10':
        if args.model_name == 'mlpnet':
            config = mlpnet_hyperparams.config
        elif args.model_name == 'vgg11_nobias':
            config = cifar10_vgg_hyperparams.config
        elif args.model_name == 'vgg11_half_nobias':
            config = cifar10_vgg_hyperparams_half.config
        elif args.model_name == 'vgg11_doub_nobias':
            config = cifar10_vgg_hyperparams_doub.config
        elif args.model_name == 'vgg11_quad_nobias':
            config = cifar10_vgg_hyperparams_quad.config
        elif args.model_name == 'resnet18_nobias':
            config = cifar10_resnet18_nobias_hyperparams.config
        elif args.model_name == 'resnet18_nobias_nobn':
            config = cifar10_resnet18_nobias_nobn_hyperparams.config
        else:
            raise NotImplementedError

    if args.second_model_name is not None:
        if 'vgg' in args.second_model_name:
            if 'half' in args.second_model_name:
                second_config = cifar10_vgg_hyperparams_half.config
            elif 'doub' in args.second_model_name:
                second_config = cifar10_vgg_hyperparams_doub.config
            elif 'quad' in args.second_model_name:
                second_config = cifar10_vgg_hyperparams_quad.config
            elif args.second_model_name == 'vgg11_nobias':
                second_config = cifar10_vgg_hyperparams.config
            else:
                raise NotImplementedError
        elif 'resnet' in args.second_model_name:
            if args.second_model_name == 'resnet18_nobias':
                second_config= cifar10_resnet18_nobias_hyperparams.config
            elif args.second_model_name == 'resnet18_nobias_nobn':
                config = cifar10_resnet18_nobias_nobn_hyperparams.config
            else:
                raise  NotImplementedError
    else:
        second_config = config

    return config, second_config

def isnan(x):
    return x != x