import torch
import torch.nn.functional as F

# GM stands for graph matching
class Ground_Metric_GM:

    def __init__( 
        self, 
        model_1_param:torch.tensor=None, 
        model_2_param:torch.tensor=None, 
        conv_param:bool=False,
        bias_param:bool=False,
        pre_conv_param:bool=False,
        pre_conv_image_size_squared:int=None ):
        '''
        if [bias_param] is True, the input parameters should be 1-d tensor of the size
            [cur_neuron_num]. Else:

        if [conv_param] is False, the input parameters should be 2-d tensors of the size
            [cur_neuron_num] x [pre_neuron_num], where each entity is simply a float

        if [conv_param] is True, the input parameters should be 3-d tensors of the size
            [cur_neuron_num] x [pre_neuron_num] x [kernel_size_squared]
        '''
        self._sanity_check( 
            model_1_param, model_2_param, conv_param, bias_param, 
            pre_conv_param, pre_conv_image_size_squared )
        '''
        if [pre_conv_param] is False:
            transforms the parameters so that the ultimate params are of the size
            1 x [cur_neuron_num * pre_neuron_num] x [edge_weight_length],
            where [edge_weight_length] is 1 if conv_param=False and [kernel_size_squared] otherwise
        elif [conv_param] is False and [bias_param] is False:
            transforms the parameters so that the ultimate params are of the size
            1 x [cur_neuron_num * pre_neuron_num / pre_kernel_size] x [pre_kernel_size]

        after the transformation, the weights in the middle dimension would be:
            0~0, 1~0, ..., N-1~0, ..., 0~M-1, 1~M-1, ..., N-1~M-1
            which is the order of edges
            ( N stands for [pre_neuron_num], and M stands for [cur_neuron_num] )
        '''
        self.model_1_param = model_1_param
        self.model_2_param = model_2_param
        self.conv_param = conv_param
        self.bias_param = bias_param
        # bias, or fully-connected from linear
        if bias_param is True or (conv_param is False and pre_conv_param is False):
            self.model_1_param = self.model_1_param.reshape( 1, -1, 1 )
            self.model_2_param = self.model_2_param.reshape( 1, -1, 1 )
        # fully-connected from conv
        elif conv_param is False and pre_conv_param is True:
            self.model_1_param = self.model_1_param.reshape( 1, -1, pre_conv_image_size_squared )
            self.model_2_param = self.model_2_param.reshape( 1, -1, pre_conv_image_size_squared )
        # conv
        else:
            self.model_1_param = self.model_1_param.reshape( 1, -1, model_1_param.shape[-1] )
            self.model_2_param = self.model_2_param.reshape( 1, -1, model_2_param.shape[-1] )
    
    def _sanity_check( 
        self,
        model_1_param:torch.tensor, 
        model_2_param:torch.tensor, 
        conv_param:bool, 
        bias_param:bool,
        pre_conv_param:bool,
        pre_conv_image_size_squared:int ):
        assert model_1_param is not None
        assert model_2_param is not None
        if bias_param is True:
            assert len( model_1_param.shape ) == 1
        elif conv_param is True:
            assert len( model_1_param.shape ) == 3
        else:
            assert len( model_1_param.shape ) == 2
        if pre_conv_param is True:
            assert type(pre_conv_image_size_squared) == int
        assert model_1_param.shape == model_2_param.shape

    def process_distance( self, p:int=2 ):
        '''
        returns the p-norm pair-wise distance between [self.model_1_param] and
            [self.model_2_param].
            
            if [bias_param] is False
                the returned value is a tensor of the size 
                [cur_neuron_num * pre_neuron_num] x [cur_neuron_num * pre_neuron_num], such that
                ` r[i][j] = the distance between edge i in model 1 and edge j in model 2 `
                ( edge i stands for the weight between neuron a in the previous layer and neuron
                b in the current layer, where ` i = a + b*N ` )
            else if [bias_param] is True
                the returned value is a tensor of the size
                [cur_neuron_num] x [cur_neuron_num]
        '''
        return torch.cdist( 
            self.model_1_param.to(torch.float), 
            self.model_2_param.to(torch.float), 
            p=p )[0]
    
    def process_soft_affinity( self, p:int=2 ):
        '''
        returns the p-norm pair-wise soft affinity between [self.model_1_param] and 
            [self.model_2_param]. 
            
            if [bias_param] is False
                the returned value [r] is a tensor of the size 
                [cur_neuron_num * pre_neuron_num] x [cur_neuron_num * pre_neuron_num], such that
                ` r[i][j] = the affinity between edge i in model 1 and edge j in model 2 `
                ( edge i stands for the weight between neuron a in the previous layer and neuron
                b in the current layer, where ` i = a + b*N ` )
            else if [bias_param] is True
                the returned value is of the size
                [cur_neuron_num] x [cur_neuron_num]
        '''
        return torch.exp( 0 - self.process_distance( p=p ) )


class Ground_Metric_GM_new:

    def __init__(
            self,
            model_1_param: torch.tensor = None,
            model_2_param: torch.tensor = None,
            conv_param: bool = False,
            bias_param: bool = False,
            pre_conv_param: bool = False,
            pre_conv_image_size_squared: int = None):
        '''
        if [bias_param] is True, the input parameters should be 1-d tensor of the size
            [cur_neuron_num]. Else:

        if [conv_param] is False, the input parameters should be 2-d tensors of the size
            [cur_neuron_num] x [pre_neuron_num], where each entity is simply a float

        if [conv_param] is True, the input parameters should be 3-d tensors of the size
            [cur_neuron_num] x [pre_neuron_num] x [kernel_size_squared]
        '''
        self._sanity_check(
            model_1_param, model_2_param, conv_param, bias_param,
            pre_conv_param, pre_conv_image_size_squared)
        '''
        if [pre_conv_param] is False:
            transforms the parameters so that the ultimate params are of the size
            1 x [cur_neuron_num * pre_neuron_num] x [edge_weight_length],
            where [edge_weight_length] is 1 if conv_param=False and [kernel_size_squared] otherwise
        elif [conv_param] is False and [bias_param] is False:
            transforms the parameters so that the ultimate params are of the size
            1 x [cur_neuron_num * pre_neuron_num / pre_kernel_size] x [pre_kernel_size]

        after the transformation, the weights in the middle dimension would be:
            0~0, 1~0, ..., N-1~0, ..., 0~M-1, 1~M-1, ..., N-1~M-1
            which is the order of edges
            ( N stands for [pre_neuron_num], and M stands for [cur_neuron_num] )
        '''
        self.model_1_param = model_1_param
        self.model_2_param = model_2_param
        self.conv_param = conv_param
        self.bias_param = bias_param
        self.shape_1 = self.model_1_param.shape[0]
        self.shape_2 = self.model_1_param.shape[1]
        # bias, or fully-connected from linear
        # if bias_param is True or (conv_param is False and pre_conv_param is False):
        #     self.model_1_param = self.model_1_param.reshape(1, -1, 1)
        #     self.model_2_param = self.model_2_param.reshape(1, -1, 1)
        # # fully-connected from conv
        # elif conv_param is False and pre_conv_param is True:
        #     self.model_1_param = self.model_1_param.reshape(1, -1, pre_conv_image_size_squared)
        #     self.model_2_param = self.model_2_param.reshape(1, -1, pre_conv_image_size_squared)
        # # conv
        # else:
        #     self.model_1_param = self.model_1_param.reshape(1, -1, model_1_param.shape[-1])
        #     self.model_2_param = self.model_2_param.reshape(1, -1, model_2_param.shape[-1])
        self.model_1_param = self.model_1_param.reshape(1, model_1_param.shape[0], -1)
        self.model_2_param = self.model_2_param.reshape(1, model_2_param.shape[0], -1)

    def _sanity_check(
            self,
            model_1_param: torch.tensor,
            model_2_param: torch.tensor,
            conv_param: bool,
            bias_param: bool,
            pre_conv_param: bool,
            pre_conv_image_size_squared: int):
        assert model_1_param is not None
        assert model_2_param is not None
        if bias_param is True:
            assert len(model_1_param.shape) == 1
        elif conv_param is True:
            assert len(model_1_param.shape) == 3
        else:
            assert len(model_1_param.shape) == 2
        if pre_conv_param is True:
            assert type(pre_conv_image_size_squared) == int
        assert model_1_param.shape == model_2_param.shape

    def process_distance(self, p: int = 2):
        '''
        returns the p-norm pair-wise distance between [self.model_1_param] and
            [self.model_2_param].
            
            if [bias_param] is False
                the returned value is a tensor of the size 
                [cur_neuron_num * pre_neuron_num] x [cur_neuron_num * pre_neuron_num], such that
                ` r[i][j] = the distance between edge i in model 1 and edge j in model 2 `
                ( edge i stands for the weight between neuron a in the previous layer and neuron
                b in the current layer, where ` i = a + b*N ` )
            else if [bias_param] is True
                the returned value is a tensor of the size
                [cur_neuron_num] x [cur_neuron_num]
        '''
        dist = torch.cdist(
            self.model_1_param.to(torch.float),
            self.model_2_param.to(torch.float),
            p=p)
        # return dist[0].reshape(self.shape_1, self.shape_2, self.shape_1, self.shape_2).mean(axis=(1, 3))

        return dist[0]

    def process_soft_affinity(self, p: int = 2):
        '''
        returns the p-norm pair-wise soft affinity between [self.model_1_param] and 
            [self.model_2_param]. 
            
            if [bias_param] is False
                the returned value [r] is a tensor of the size 
                [cur_neuron_num * pre_neuron_num] x [cur_neuron_num * pre_neuron_num], such that
                ` r[i][j] = the affinity between edge i in model 1 and edge j in model 2 `
                ( edge i stands for the weight between neuron a in the previous layer and neuron
                b in the current layer, where ` i = a + b*N ` )
            else if [bias_param] is True
                the returned value is of the size
                [cur_neuron_num] x [cur_neuron_num]
        '''
        return torch.exp(0 - 0.5 * self.process_distance(p=p))

if __name__ == '__main__':
    print( '---------- testing on fully-connected layers weights ----------' )
    param_1 = torch.tensor( 
        [   [1, 2, 3, 4],
            [5, 6, 7, 8]   ] )
    param_2 = torch.tensor(
        [   [2, 4, 6, 8],
            [3, 5, 7, 9]   ] )
    gm = Ground_Metric_GM( param_1, param_2, conv_param=False, bias_param=False )
    print( f'\tdistance is \n\t{gm.process_distance()},\n\tsoft_affinity is \n\t{gm.process_soft_affinity()}' )

    print( '---------- testing on convlutional layers weights ----------' )
    param_1 = torch.tensor( 
        [   [[1,2,3,4], [1,2,3,4], [1,2,3,4], [1,2,3,4]],
            [[1,2,3,4], [1,2,3,4], [1,2,3,4], [1,2,3,4]]   ] )
    param_2 = torch.tensor(
        [   [[1,2,3,4], [5,6,7,8], [1,2,3,4], [5,6,7,8]],
            [[1,2,3,4], [5,6,7,8], [1,2,3,4], [5,6,7,8]]   ] )
    gm = Ground_Metric_GM( param_1, param_2, conv_param=True, bias_param=False )
    print( f'\tdistance is \n\t{gm.process_distance()},\n\tsoft_affinity is \n\t{gm.process_soft_affinity()}' )

    print( '---------- testing on layers biases ----------' )
    param_1 = torch.tensor([1,2,3,4,5,6,7,8])
    param_2 = torch.tensor([8,7,6,5,4,3,2,1])
    gm = Ground_Metric_GM( param_1, param_2, conv_param=False, bias_param=True )
    print( f'\tdistance is \n\t{gm.process_distance()},\n\tsoft_affinity is \n\t{gm.process_soft_affinity()}' )