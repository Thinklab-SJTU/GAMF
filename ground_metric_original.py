import torch
from utils_gm import isnan

class GroundMetric:
    """
        Ground Metric object for Wasserstein computations:
    """

    def __init__(self, params, not_squared = False):
        self.params = params
        self.ground_metric_type = params.ground_metric
        self.ground_metric_normalize = params.ground_metric_normalize
        self.reg = params.reg   # what is reg?
        if hasattr(params, 'not_squared'):
            self.squared = not params.not_squared
        else:
            # so by default squared will be on!
            self.squared = not not_squared
        self.mem_eff = params.ground_metric_eff     # what is mem_eff?

    def _clip(self, ground_metric_matrix):
        '''
            clip (namely set a upper boundary to all entries in the matrix and just cut
            away the part that exceeds that boundary) the matrix
        '''

        if self.params.debug:
            print("before clipping", ground_metric_matrix.data)

        percent_clipped = (float((ground_metric_matrix >= self.reg * self.params.clip_max).long().sum().data) \
                           / ground_metric_matrix.numel()) * 100    # what is numel()?
        print("percent_clipped is (assumes clip_min = 0) ", percent_clipped)
        setattr(self.params, 'percent_clipped', percent_clipped)
        # will keep the M' = M/reg in range clip_min and clip_max
        ground_metric_matrix.clamp_(min=self.reg * self.params.clip_min,
                                             max=self.reg * self.params.clip_max)
        if self.params.debug:
            print("after clipping", ground_metric_matrix.data)
        return ground_metric_matrix

    def _normalize(self, ground_metric_matrix):
        '''
            normalize the ground metric matrix by different methods including
            log, max, median, mean.
        '''

        if self.ground_metric_normalize == "log":
            ground_metric_matrix = torch.log1p(ground_metric_matrix)
        elif self.ground_metric_normalize == "max":
            print("Normalizing by max of ground metric and which is ", ground_metric_matrix.max())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.max()
        elif self.ground_metric_normalize == "median":
            print("Normalizing by median of ground metric and which is ", ground_metric_matrix.median())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.median()
        elif self.ground_metric_normalize == "mean":
            print("Normalizing by mean of ground metric and which is ", ground_metric_matrix.mean())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.mean()
        elif self.ground_metric_normalize == "none":
            return ground_metric_matrix
        else:
            raise NotImplementedError

        return ground_metric_matrix

    def _sanity_check(self, ground_metric_matrix):
        assert not (ground_metric_matrix < 0).any()
        assert not (isnan(ground_metric_matrix).any())

    def _cost_matrix_xy(self, x, y, p=2, squared = True):
        '''
            this function calculates the cost metric between two neuron by their
            weights. This is the exact function that we need to modify
            this function is exactly the same as [_get_pairwise_distance()] if [p] == 2

            an example to understand the code:
            let xw1 = [1,2,3,4], xw2 = [2,3,4,5]
                yw1 = [3,4,5,6], yw2 = [4,5,6,7]
            x = torch.tensor([ [1,2,3,4], [2,3,4,5] ]) = [ xw1, xw2 ]
            y = torch.tensor([ [3,4,5,6], [4,5,6,7] ]) = [ yw1, yw2 ]
            x_col = tensor([[ [1,2,3,4] ],
                            [ [2,3,4,5] ]])            = [ [xw1], [xw2] ]
            y_lin = tensor([[ [3,4,5,6], 
                              [4,5,6,7] ]])            = [ [ yw1, yw2 ] ]
            x_col - y_lin = tensor([[ [2, 2, 2, 2],
                                      [3, 3, 3, 3] ],

                                    [ [1, 1, 1, 1],
                                      [2, 2, 2, 2] ]]) = [ [ xw1-yw1, xw1-yw2 ], 
                                                           [ xw2-yw1, xw2-yw2 ] ]
            torch.sum( torch.abs(x_col - y_lin) ** p, 2 )
                          = [ [ ||xw1-yw1||, ||xw1-yw2|| ],
                              [ ||xw2-yw1||, ||xw2-yw2|| ] ]
        '''

        # TODO: Use this to guarantee reproducibility of previous results and then move onto better way
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
        if not squared:
            # Alexanderia
            # print("dont leave off the squaring of the ground metric")
            c = c ** (1/2)
        # print(c.size())
        if self.params.dist_normalize:
            assert NotImplementedError
        return c


    def _pairwise_distances(self, x, y=None, squared=True):
        '''
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        dist = torch.clamp(dist, min=0.0)

        if self.params.activation_histograms and self.params.dist_normalize:
            dist = dist/self.params.act_num_samples
            print("Divide squared distances by the num samples")

        if not squared:
            # Alexanderia
            # print("dont leave off the squaring of the ground metric")
            dist = dist ** (1/2)

        return dist

    def _get_euclidean(self, coordinates, other_coordinates=None):
        # TODO: Replace by torch.pdist (which is said to be much more memory efficient)

        if other_coordinates is None:
            matrix = torch.norm(
                coordinates.view(coordinates.shape[0], 1, coordinates.shape[1]) \
                - coordinates, p=2, dim=2
            )
        else:
            if self.mem_eff:
                matrix = self._pairwise_distances(coordinates, other_coordinates, squared=self.squared)
            else:
                matrix = self._cost_matrix_xy(coordinates, other_coordinates, squared = self.squared)

        return matrix

    def _normed_vecs(self, vecs, eps=1e-9):
        norms = torch.norm(vecs, dim=-1, keepdim=True)
        print("stats of vecs are: mean {}, min {}, max {}, std {}".format(
            norms.mean(), norms.min(), norms.max(), norms.std()
        ))
        return vecs / (norms + eps)

    def _get_cosine(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            matrix = coordinates / torch.norm(coordinates, dim=1, keepdim=True)
            matrix = 1 - matrix @ matrix.t()    # "@" symbol means matrix multiplication
        else:
            matrix = 1 - torch.div(
                coordinates @ other_coordinates.t(),
                torch.norm(coordinates, dim=1).view(-1, 1) @ torch.norm(other_coordinates, dim=1).view(1, -1)
            )
        return matrix.clamp_(min=0)

    def _get_angular(self, coordinates, other_coordinates=None):
        pass

    def get_metric(self, coordinates, other_coordinates=None):
        '''
        if [other_coordinates] is None, then only calculates the distances among the vectors 
            in [coordinates] only
        otherwise, calculate the distances among the vectors in [coordinates] and [other_coordinates]
        '''
        get_metric_map = {
            'euclidean': self._get_euclidean,
            'cosine': self._get_cosine,
            'angular': self._get_angular,
        }
        return get_metric_map[self.ground_metric_type](coordinates, other_coordinates)

    def process(self, coordinates, other_coordinates=None):
        '''
        1. get the metric matrix
        2. normalize the metric matrix
        3. clip if required
        4. return the matrix
        '''
        # Alexanderia
        # print('Processing the coordinates to form ground_metric')
        if self.params.geom_ensemble_type == 'wts' and self.params.normalize_wts:
            print("In weight mode: normalizing weights to unit norm")
            coordinates = self._normed_vecs(coordinates)
            if other_coordinates is not None:
                other_coordinates = self._normed_vecs(other_coordinates)

        ground_metric_matrix = self.get_metric(coordinates, other_coordinates)

        if self.params.debug:
            print("coordinates is ", coordinates)
            if other_coordinates is not None:
                print("other_coordinates is ", other_coordinates)
            print("ground_metric_matrix is ", ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        ground_metric_matrix = self._normalize(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        if self.params.clip_gm:
            ground_metric_matrix = self._clip(ground_metric_matrix)

        self._sanity_check(ground_metric_matrix)

        if self.params.debug:
            print("ground_metric_matrix at the end is ", ground_metric_matrix)

        return ground_metric_matrix
