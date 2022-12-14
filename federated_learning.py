import torch
from torch import optim
from torch.nn import parameter
import torchvision
import argparse
import copy
import time

import model_gm as model
import data_gm as data
import fusion_gm as fusion
import fusion_gm_slice as fusion_slice
import wasserstein_ensemble_gm as ot

def get_config():
    '''
    get the configurations from commandline with parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument( '--num_epoch', default=48, type=int, 
                        help='number of epochs (default 48)' )
    parser.add_argument( '--batch_size', default=50, type=int, 
                        help='size of each mini batch (default 50)' )
    parser.add_argument( '--learning_rate', default=0.000075, type=float, 
                        help='learning rate (default 0.000075)' )
    parser.add_argument( '--epoch_per_averaging', default=1, type=int, 
                        help='number of batches per averaging (default 1)' )
    parser.add_argument( '--model_name', default='simplemnistnet', type=str, 
                        help='the name of the model (default simplemnistnet)' )
    parser.add_argument( '--device', default='cpu', type=str, 
                        help='the device to train the model, either cpu or cuda (default cpu)' )
    parser.add_argument( '--dataset', default='mnist', type=str, 
                        help='the dataset to train the model (default mnist)' )
    parser.add_argument( '--need_customized_dataset', default=False, type=bool, 
                        help='whether or not you need customized dataset (default: False)' )
    parser.add_argument( '--iid', default=True, type=bool, 
                        help='whether or not you need iid dataset or not (default: True)' )
    parser.add_argument( '--to_download', default=True, type=bool, 
                        help='whether or not the dataset needs to be downloaded (default: True)' )
    parser.add_argument( '--ensemble_step', default=0.5, type=float, 
                        help='the ensemble weight used in gm-based fusion (default: 0.5)' )
    parser.add_argument( '--training_mode', default='traditional', type=str, 
                        help='whether to use traditional averaging or fusion-based averaging, \
                            can be traditional or fusion or fusion_slice or ot (default: traditional)' )

    args = parser.parse_args()
    if args.device not in ['cpu', 'cuda']:
        raise NotImplementedError
    args.device = torch.device( 'cpu' ) if args.device == 'cpu' else torch.device( 'cuda' )

    args.batch_size_train = args.batch_size
    args.batch_size_test = args.batch_size


    args.act_num_samples = 200
    args.clip_gm = False
    args.clip_max = 5
    args.clip_min = 0
    args.correction = True
    args.dataset = "mnist"
    args.debug = False
    args.dist_normalize = True
    args.ensemble_step = 0.5
    args.eval_aligned = False
    args.exact = True
    args.geom_ensemble_type = "wts"
    args.gpu_id = -1
    args.ground_metric = "euclidean"
    args.ground_metric_eff = False
    args.ground_metric_normalize = "none"
    args.importance = None
    args.normalize_wts = False
    args.num_models = 2
    args.not_squared = True
    args.past_correction = True
    args.prediction_wts = True
    args.proper_marginals = False
    args.reg = 0.01
    args.skip_last_layer = False
    args.softmax_temperature = 1
    args.unbalanced = False
    args.weight = [0.5, 0.5]
    args.width_ratio = 1

    return args

class federated_learning:
    def __init__( self, args, train_set=None, test_set=None ):
        '''
        initialize two models and the corresponding datasets
        '''
        self.model1 = model.get_model_from_name( args )
        self.model2 = model.get_model_from_name( args )
        # TODO: add training sets here
        self.train_loader_1, self.train_loader_2, self.test_loader = \
                    data.get_federated_data_loader( args, [0.5, 0.5], non_iid=not args.iid )
        self.config = copy.deepcopy( args )
    
    def _train_model_i_step( self, idx ):
        '''
        train model [idx]. Currently, [idx] is either 1 or 2
        '''
        target_model = self.model1 if idx == 1 else self.model2
        target_model_parameters = self.model1.parameters() if idx == 1 else self.model2.parameters()
        target_train_loader = self.train_loader_1 if idx == 1 else self.train_loader_2
        optimizer = torch.optim.SGD( target_model_parameters, lr=self.config.learning_rate )
        criterion = torch.nn.CrossEntropyLoss()

        print( f'one step training on model {idx}' )
        for epoch_idx in range( self.config.epoch_per_averaging ):
            for i, ( data, label ) in enumerate( target_train_loader ):
                data = data.to( self.config.device )
                label = label.to( self.config.device )
                
                y_pred = target_model( data )
                loss = criterion( y_pred, label )

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                if ( i + 1 ) % 100 == 0:
                    print( f'epoch {epoch_idx + 1} / {self.config.epoch_per_averaging}, \
                        step {i + 1} / {len( target_train_loader )}, \
                        loss = {loss.item():.4f}' )

    def _federated_averaging_traditional( self ):
        '''
        average the parameters of two models based on vanilla averaging method
        '''
        state_dict_1 = self.model1.state_dict()
        state_dict_2 = self.model2.state_dict()
        for name, _ in state_dict_1.items():
            temp = (state_dict_1[name] + state_dict_2[name]) / 2
            state_dict_1[name] = temp
            state_dict_2[name] = temp
        print( f'averaging parameters based on vanilla averaging' )
        self.model1.load_state_dict( state_dict_1 )
        self.model2.load_state_dict( state_dict_2 )

    def _federated_averaging_fusion( self ):
        '''
        average the parameters of two models based on gm_based_fusion method
        '''
        # the following code can be replaced by other functions
        parameters, _ = fusion.graph_matching_fusion( 
                                    self.config, [self.model1, self.model2] )
        
        state_dict_1 = self.model1.state_dict()
        state_dict_2 = self.model2.state_dict()
        for idx, (name, _) in enumerate( state_dict_1.items() ):
            state_dict_1[name] = parameters[idx]
            state_dict_2[name] = parameters[idx]
        print( f'averaging parameters based on gm_based fusion' )
        self.model1.load_state_dict( state_dict_1 )
        self.model2.load_state_dict( state_dict_2 )
    
    def _federated_averaging_fusion_slice( self ):
        '''
        average the parameters of two models based on gm_based_fusion method
        '''
        # the following code can be replaced by other functions
        parameters, _ = fusion_slice.graph_matching_fusion_slice( 
                                    self.config, [self.model1, self.model2] )
        
        state_dict_1 = self.model1.state_dict()
        state_dict_2 = self.model2.state_dict()
        for idx, (name, _) in enumerate( state_dict_1.items() ):
            state_dict_1[name] = parameters[idx]
            state_dict_2[name] = parameters[idx]
        print( f'averaging parameters based on gm_based fusion' )
        self.model1.load_state_dict( state_dict_1 )
        self.model2.load_state_dict( state_dict_2 )
    
    def _federated_averaging_ot( self ):
        '''
        average the parameters of two models based on ot_based_fusion method
        '''
        # the following code can be replaced by other functions
        parameters, _ = ot.get_wassersteinized_layers_modularized( 
                                    self.config, [self.model1, self.model2] )
        
        state_dict_1 = self.model1.state_dict()
        state_dict_2 = self.model2.state_dict()
        for idx, (name, _) in enumerate( state_dict_1.items() ):
            state_dict_1[name] = parameters[idx]
            state_dict_2[name] = parameters[idx]
        print( f'averaging parameters based on gm_based fusion' )
        self.model1.load_state_dict( state_dict_1 )
        self.model2.load_state_dict( state_dict_2 )
    
    def federated_train( self ):
        start_time = time.perf_counter()
        for epoch_idx in range( int( self.config.num_epoch / self.config.epoch_per_averaging ) + 1 ):
            print( f'------ federated learning iteration {epoch_idx} -------' )
            self._train_model_i_step( 1 )
            self._train_model_i_step( 2 )
            if self.config.training_mode == 'traditional':
                self._federated_averaging_traditional()
            elif self.config.training_mode == 'ot':
                self._federated_averaging_ot()
            elif self.config.training_mode == 'fusion_slice':
                self._federated_averaging_fusion_slice()
            else:
                self._federated_averaging_fusion()
        end_time = time.perf_counter()
        print( f'time consumed for training is {end_time - start_time}' )

    def test( self ):
        '''
        test the accuracy of the model 1 (which is actually the overall model after fusion)
        '''
        print( f'begin to test the model' )
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for data, label in self.test_loader:
                data = data.to( self.config.device )
                label = label.to( self.config.device )
                outputs = self.model1( data )
                pred = outputs.data.max(1, keepdim=True)[1]
                n_correct += pred.eq(label.data.view_as(pred)).sum()
        print('\nTest results: Accuracy: {}/{} ({:.0f}%)\n'.format(
            n_correct, len(self.test_loader.dataset),
            100. * n_correct / len(self.test_loader.dataset)))


if __name__ == '__main__':
    args = get_config()
    fl_client = federated_learning( args )
    
    fl_client.federated_train()
    fl_client.test()