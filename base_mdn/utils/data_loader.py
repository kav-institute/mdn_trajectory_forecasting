from __future__ import print_function
from six.moves import cPickle as pickle
from termcolor import colored
from utils.helper import generate_unique_randoms

import numpy as np
import random


class DataLoader:
    """Trajectory dataset data loader class
    """
    
    def __init__(self, cfg=''):
        """Init and setup
        """
        
        # variables
        self.name = cfg.name
        self.train_data_paths = cfg.paths['train_data_paths']
        self.eval_data_paths = cfg.paths['val_data_paths']
        self.test_data_paths = cfg.paths['test_data_paths']
        self.randomize = cfg.train_params['randomize_train_data']
        self.batch_reduction = cfg.train_params['batch_reduction']
        self.with_print = cfg.with_print
        
        # data storages
        self.train_data = []
        self.eval_data = []
        self.test_data = []
        
        return
        
        
    def load_train_data(self):
        """Load training data from file
        """
        
        if self.with_print: print(colored(f" Data loader: loading train data: {self.train_data_paths}", 'green'))
        self.train_data = self.load_pkl_track(paths=self.train_data_paths)
        if self.with_print: print(colored(f" Data loader: loaded {len(self.train_data[0])} samples for training", 'green'))
        if self.with_print: print(colored(f" with randomized training data: {self.randomize}", 'green'))
        if self.with_print: print(colored(f" batch reduction scale: {self.batch_reduction}", 'green'))
        return
        
        
    def load_eval_data(self):
        """Load train-eval data from file
        """
        
        if self.with_print: print(colored(f" Data loader: loading eval data: {self.eval_data_paths}", 'green'))
        self.eval_data = self.load_pkl_all(paths=self.eval_data_paths)
        if self.with_print: print(colored(f" Data loader: loaded {len(self.eval_data[0])} samples for train-eval", 'green'))
        if self.with_print: print(colored(f" batch reduction scale: {self.batch_reduction}", 'green'))
        return
        
        
    def load_test_data(self):
        """Load test data from file
        """
        
        if self.with_print: print(colored(f" Data loader: loading test data: {self.test_data_paths}", 'green'))
        self.test_data = self.load_pkl_all(paths=self.test_data_paths)
        if self.with_print: print(colored(f" Data loader: loaded {len(self.test_data[0])} samples for testing", 'green'))
        return
    
    
    def get_train_data(self):
        """Get training data for next epoch
        """
        
        # with random suffle
        if self.randomize:
            
            shuffled_indices = list(range(0, len(self.train_data[0])))
            random.shuffle(shuffled_indices)
            self.train_data[0] = np.take(a=self.train_data[0], indices=shuffled_indices, axis=0)
            self.train_data[1] = np.take(a=self.train_data[1], indices=shuffled_indices, axis=0)
            
        # reduce train data to random subset
        if self.batch_reduction < 1.0:
            
            n = int(len(self.train_data[0]) * self.batch_reduction)
            random_reduced_indices = generate_unique_randoms(count=n, min_value=0, max_value=len(self.train_data[0])-1)
            X = np.take(a=self.train_data[0], indices=random_reduced_indices, axis=0)
            y = np.take(a=self.train_data[1], indices=random_reduced_indices, axis=0)
            
            return X, y[:,:,:2]
        
        # with no modifications
        else:
            
            return self.train_data[0], self.train_data[1][:,:,:2]
    
    
    def get_eval_data(self):
        """Get eval data for next epoch
        """
        
        # reduce train data to random subset
        if self.batch_reduction < 1.0:
            
            n = int(len(self.eval_data[0]) * self.batch_reduction)
            random_reduced_indices = generate_unique_randoms(count=n, min_value=0, max_value=len(self.eval_data[0])-1)
            X = np.take(a=self.eval_data[0], indices=random_reduced_indices, axis=0)
            y = np.take(a=self.eval_data[1], indices=random_reduced_indices, axis=0)
            p = np.take(a=self.eval_data[2], indices=random_reduced_indices, axis=0)
            r = np.take(a=self.eval_data[3], indices=random_reduced_indices, axis=0)
            s = np.take(a=self.eval_data[4], indices=random_reduced_indices, axis=0)
            
            return X[:,:,:2], y[:,:,:2], p, r, s
        
        # with no modifications
        else:
            
            return self.eval_data[0], self.eval_data[1][:,:,:2], self.eval_data[2], self.eval_data[3], self.eval_data[4]
        
    
    def get_test_data(self):
        """Get test data for next epoch
        """
        
        return self.test_data[0], self.test_data[1][:,:,:2], self.test_data[2], self.test_data[3], self.test_data[4]
    
    
    def reorder_list(lst, indices):
        """Create a reordered copy of a given list
        """
        
        # create an empty list to store the elements in their new order
        tmp = []
        
        # iterate over the indices and add the corresponding elements from the original list to the new list
        for index in indices:
            
            if 0 <= index < len(lst):
                
                tmp.append(lst[index])
                
        return tmp
    
    
    def load_pkl_track(self, paths):
        """Load track data from a .pkl file
        """
        
        nX = []
        ny = []
        
        for p in paths:
        
            with open(p, 'rb') as f:
                ego_data = pickle.load(f)
            
            for id in ego_data:
                
                nX.append(ego_data[id]['X'])
                ny.append(ego_data[id]['y'])
                
        return [np.array(nX), np.array(ny)]
    
    
    def load_pkl_all(self, paths):
        """Load track and transformation data from a .pkl file
        """
        
        nX = []
        ny = []
        npos = []
        nrot = []
        nsrc = []
        
        for p in paths:
        
            with open(p, 'rb') as f:
                ego_data = pickle.load(f)
                
            for id in ego_data:
                
                nX.append(ego_data[id]['X'])
                ny.append(ego_data[id]['y'])
                npos.append(ego_data[id]['reference_position'])
                nrot.append(ego_data[id]['rotation_angle'])
                nsrc.append(ego_data[id]['source'])
                
        return [np.array(nX), np.array(ny), np.array(npos), np.array(nrot), np.array(nsrc)]