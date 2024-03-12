import os
import json
import copy
import seaborn as sns
import numpy as np
from termcolor import colored


class ConfigLoader:
    """ Class loading parameters and and paths from external config file
    """
    
    def __init__(self, config_path, target, with_log, with_print, name, model_arch, type):
        """ Init and setup 
        """
        
        # load config from file
        if with_print: print(colored(f"Loading config data for config: {name}", 'green'))
        
        with open(config_path) as f:
            cfg = json.load(f)
        
        # variables and paths
        self.name = name
        self.target = target
        self.model_arch = model_arch
        self.type = type
        self.with_log = with_log
        self.with_print = with_print
        self.paths = cfg['paths']
        self.model_params = cfg['model_params']
        self.train_params = cfg['train_params']
        self.test_params = cfg['test_params']
        self.eval_metrics = cfg['eval_metrics']
        
        # eval and plotting
        self.reliability_bins = [k for k in np.arange(0.0, 1.01, 0.01)]
        palette = np.array(sns.color_palette(palette='deep', n_colors=len(self.test_params['test_horizons'])))
        self.colors_rgb = copy.deepcopy(palette)
        self.colors_bgr = copy.deepcopy(palette)
        self.colors_bgr[:, [2, 0]] = self.colors_bgr[:, [0, 2]]
        
        # paths
        self.result_path = os.path.join(self.paths['result_path'], self.model_arch, target, self.name)
        self.checkpoint_path = os.path.join(self.result_path, 'checkpoints')
        self.evaluation_path = os.path.join(self.result_path, 'evaluation')
        self.eval_ego_examples_path = os.path.join(self.result_path, 'evaluation', 'examples', 'ego')
        self.eval_world_examples_path = os.path.join(self.result_path, 'evaluation', 'examples', 'world')
        self.testing_path = os.path.join(self.result_path, 'testing')
        self.test_ego_examples_path = os.path.join(self.result_path, 'testing', 'examples', 'ego')
        self.test_world_examples_path = os.path.join(self.result_path, 'testing', 'examples', 'world')
        
        # create result dirs
        if not os.path.exists(self.result_path): os.makedirs(self.result_path) 
        if not os.path.exists(self.checkpoint_path): os.makedirs(self.checkpoint_path)
        if not os.path.exists(self.evaluation_path): os.makedirs(self.evaluation_path)
        if not os.path.exists(self.eval_ego_examples_path): os.makedirs(self.eval_ego_examples_path)
        if not os.path.exists(self.eval_world_examples_path): os.makedirs(self.eval_world_examples_path)
        if not os.path.exists(self.testing_path): os.makedirs(self.testing_path)
        if not os.path.exists(self.test_ego_examples_path): os.makedirs(self.test_ego_examples_path)
        if not os.path.exists(self.test_world_examples_path): os.makedirs(self.test_world_examples_path)
