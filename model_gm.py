import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
PATH_TO_CIFAR = "./cifar_gm/"
sys.path.append(PATH_TO_CIFAR)
import train as cifar_train

def get_model_from_name(*, name='', args=None, idx=-1):
    '''
        [args.width_ratio] determines the portion that is remained after thining the network
            if [width_ratio] = -1, then it means no thining is applied
            else, [width_ratio] should be positive
        if [idx] is -1, then [weight_ratio] is -1, which means no width resizing in hidden layers
        Otherwise if [idx] is 0 and only one model is available, then [weight_ratio] is 
            [args.width_ratio] and width is resized
    '''
    from utils_gm import Namespace
    
    if args == None:
        args = Namespace( model_name=name )

    if idx != -1 and idx == (args.num_models - 1):
        # only passes for the second model
        width_ratio = args.width_ratio
    else:
        width_ratio = -1

    if args.model_name == 'net':
        return Net(args)
    elif args.model_name == 'simplenet':
        return SimpleAhaNet(args)
    elif args.model_name == 'smallmlpnet':
        return SmallMlpNet(args)
    elif args.model_name == 'mlpnet':
        return MlpNet(args, width_ratio=width_ratio)
    elif args.model_name == 'bigmlpnet':
        return BigMlpNet(args)
    elif args.model_name == 'cifarmlpnet':
        return CifarMlpNet(args)
    ### my models begin
    elif args.model_name == 'naivenet':
        return naive_net()
    elif args.model_name == 'naivecnn':
        return naive_cnn()
    elif args.model_name == 'simplemnistnet':
        return SimpleNet( args )
    elif args.model_name == 'lenet':
        return LeNet()
    elif args.model_name == 'smalllenet':
        return SmallLeNet()
    ### my models end
    elif args.model_name[0:3] == 'vgg' or args.model_name[0:3] == 'res':
        if args.second_model_name is None or idx == 0:
            barebone_config = {'model': args.model_name, 'dataset': args.dataset}
        else:
            barebone_config = {'model': args.second_model_name, 'dataset': args.dataset}

        # if you want pre-relu acts, set relu_inplace to False
        return cifar_train.get_model(barebone_config, args.gpu_id, relu_inplace=not args.prelu_acts)
    else:
        print( f"model name {args.model_name} not found!" )
        

class naive_net( nn.Module ):
    def __init__( self ):
        super( naive_net, self ).__init__()
        self.lin1 = nn.Linear( 2, 3 )
        self.lin2 = nn.Linear( 3, 2 )
    def forward( self, x:torch.tensor ):
        assert x.shape == torch.Size([2])
        x = self.lin1( x )
        x = F.relu( x )
        x = self.lin2( x )
        return x
    
class naive_cnn( nn.Module ):
    def __init__( self ):
        super( naive_cnn, self ).__init__()
        self.conv1 = nn.Conv2d( 1, 2, 2 )
        self.fc1 = nn.Linear( 8, 2 )
    def forward( self, x:torch.tensor ):
        x = F.relu( self.conv1( x ) )
        x = self.fc1( x )
        return x
    
class SimpleNet( nn.Module ):
    def __init__( self, args ):
        super( SimpleNet, self ).__init__()
        # self.conv1 = nn.Conv2d( 1, args.hidden_size_1_conv, args.conv_kernel_size, padding=args.conv_padding, device=args.device, bias=args.bias )
        # self.maxpool = nn.MaxPool2d( args.maxpool_kernel_size, padding=args.maxpool_padding )
        # self.conv2 = nn.Conv2d( args.hidden_size_1_conv, args.hidden_size_2_conv, args.conv_kernel_size, padding=args.conv_padding, device=args.device, bias=args.bias )
        # self.fc1 = nn.Linear( args.input_size_fc, args.num_classes, device=args.device, bias=args.bias )
        self.conv1 = nn.Conv2d( 1, 32, 5, padding='same', device=args.device, bias=False )
        self.maxpool = nn.MaxPool2d( 2, padding=0 )
        self.conv2 = nn.Conv2d( 32, 64, 5, padding='same', device=args.device, bias=False )
        self.fc1 = nn.Linear( 3136, 10, device=args.device, bias=False )
        
    def forward( self, x ):
        output = F.relu( self.conv1( x ) )
        output = self.maxpool( output )
        output = F.relu( self.conv2( output ) )
        output = self.maxpool( output )
        output = output.view( output.shape[0], -1 )
        output = self.fc1( output )
        return output

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5), bias=False)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), bias=False)
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SmallLeNet(nn.Module):
    def __init__(self):
        super(SmallLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=(5, 5), bias=False)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=(5, 5), bias=False)
        self.fc1 = nn.Linear(3 * 5 * 5, 12, bias=False)
        self.fc2 = nn.Linear(12, 8, bias=False)
        self.fc3 = nn.Linear(8, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LogisticRegressionModel(nn.Module):
    # default input and output dim for
    def __init__(self, input_dim=784, output_dim=10):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        out = F.softmax(self.linear(x))
        return out

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias= not args.disable_bias)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias= not args.disable_bias)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50, bias= not args.disable_bias)
        self.fc2 = nn.Linear(50, 10, bias= not args.disable_bias)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class SimpleAhaNet(nn.Module):
    def __init__(self, args):
        super(SimpleAhaNet, self).__init__()
        self.fc1 = nn.Linear(784, args.num_hidden_nodes, bias= not args.disable_bias)
        self.fc2 = nn.Linear(args.num_hidden_nodes, 10, bias= not args.disable_bias)
        self.enable_dropout = args.enable_dropout

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class MlpNet(nn.Module):
    def __init__(self, args, width_ratio=-1):
        super(MlpNet, self).__init__()
        if args.dataset == 'mnist':
            # 28 x 28 x 1
            input_dim = 784
        elif args.dataset.lower() == 'cifar10':
            # 32 x 32 x 3
            input_dim = 3072
        if width_ratio != -1:
            self.width_ratio = width_ratio
        else:
            self.width_ratio = 1

        self.fc1 = nn.Linear(input_dim, int(args.num_hidden_nodes1/self.width_ratio), bias=not args.disable_bias)
        self.fc2 = nn.Linear(int(args.num_hidden_nodes1/self.width_ratio), int(args.num_hidden_nodes2/self.width_ratio), bias=not args.disable_bias)
        self.fc3 = nn.Linear(int(args.num_hidden_nodes2/self.width_ratio), int(args.num_hidden_nodes3/self.width_ratio), bias=not args.disable_bias)
        self.fc4 = nn.Linear(int(args.num_hidden_nodes3/self.width_ratio), 10, bias=not args.disable_bias)
        self.enable_dropout = args.enable_dropout

    def forward(self, x, disable_logits=False):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc4(x)

        if disable_logits:
            return x
        else:
            return F.log_softmax(x)


class SmallMlpNet(nn.Module):
    def __init__(self, args):
        super(SmallMlpNet, self).__init__()
        self.fc1 = nn.Linear(784, args.num_hidden_nodes1, bias=not args.disable_bias)
        self.fc2 = nn.Linear(args.num_hidden_nodes1, args.num_hidden_nodes2, bias=not args.disable_bias)
        self.fc3 = nn.Linear(args.num_hidden_nodes2, 10, bias=not args.disable_bias)
        self.enable_dropout = args.enable_dropout

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc3(x)

        return F.log_softmax(x)


class BigMlpNet(nn.Module):
    def __init__(self, args):
        super(BigMlpNet, self).__init__()
        if args.dataset == 'mnist':
            # 28 x 28 x 1
            input_dim = 784
        elif args.dataset.lower() == 'cifar10':
            # 32 x 32 x 3
            input_dim = 3072
        self.fc1 = nn.Linear(input_dim, args.num_hidden_nodes1, bias=not args.disable_bias)
        self.fc2 = nn.Linear(args.num_hidden_nodes1, args.num_hidden_nodes2, bias=not args.disable_bias)
        self.fc3 = nn.Linear(args.num_hidden_nodes2, args.num_hidden_nodes3, bias=not args.disable_bias)
        self.fc4 = nn.Linear(args.num_hidden_nodes3, args.num_hidden_nodes4, bias=not args.disable_bias)
        self.fc5 = nn.Linear(args.num_hidden_nodes4, 10, bias=not args.disable_bias)
        self.enable_dropout = args.enable_dropout

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc4(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc5(x)

        return F.log_softmax(x)


class CifarMlpNet(nn.Module):
    def __init__(self, args):
        super(CifarMlpNet, self).__init__()
        input_dim = 3072
        self.fc1 = nn.Linear(input_dim, 1024, bias=not args.disable_bias)
        self.fc2 = nn.Linear(1024, 512, bias=not args.disable_bias)
        self.fc3 = nn.Linear(512, 128, bias=not args.disable_bias)
        self.fc4 = nn.Linear(128, 10, bias=not args.disable_bias)
        self.enable_dropout = args.enable_dropout

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        if self.enable_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc4(x)

        return F.log_softmax(x)
