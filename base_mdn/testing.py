import os
import torch
import logging

from base_lstm import LSTM_Trajectory_Forecast
from eval import MDN_Forecaster
from utils.config_loader import ConfigLoader
from utils.helper import config_parser, count_model_parameters
from utils.data_loader import DataLoader
from termcolor import colored


def testing(args, gpu_id):
    """Run testing framework
    """
    
    #--- init and setup
    model_arch = 'base_mdn'
    type = 'testing'
    
    print(colored(f"Starting testing: on GPU: {gpu_id}", 'green'))
    
    # load configs from file
    config_dir = os.path.join(os.getcwd(), 'src', model_arch, 'configs', args.target)
    
    # multiple configs
    if args.configs == 'all':
        
        configs = [ConfigLoader(config_path=os.path.join(config_dir, conf), target=args.target, with_log=args.log, with_print=args.print, name=conf, model_arch=model_arch, type=type) for conf in os.listdir(config_dir) if conf.endswith('.json')]
    
    # single config, selected by user
    else:
        
        configs = [ConfigLoader(config_path=os.path.join(config_dir, args.configs), target=args.target, with_log=args.log, with_print=args.print, name=args.configs[:-5], model_arch=model_arch, type=type)]
    
    # start the testings
    for cfg in configs:
        
        # model params
        lstm_input_shape = cfg.model_params["lstm_input_shape"]
        max_input_horizon = cfg.model_params['max_input_horizon']
        
        # training params
        batch_size = cfg.test_params['batch_size']
        
        # eval params
        confidence_levels = cfg.test_params['confidence_levels']
        mesh_range_x = cfg.test_params['mesh_range_x']
        mesh_range_y = cfg.test_params['mesh_range_y']
        mesh_resolution = cfg.test_params['mesh_resolution']
        num_samples = cfg.test_params['num_samples']
        plot_examples = cfg.test_params['plot_examples']
        plot_examples_to_map = cfg.test_params['plot_examples_to_map']
        plot_step = cfg.test_params['plot_step']
        
        # set cuda gpu device id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # init dataloader
        data_loader = DataLoader(cfg=cfg)
        
        # load data
        data_loader.load_test_data()
        
        # logger
        if cfg.with_log:
            
            log_file_path = os.path.join(cfg.testing_path, 'testing.log')
            os.remove(log_file_path) if os.path.exists(log_file_path) else None        
            test_logger = logging.getLogger('testing')
            test_logger.setLevel(logging.INFO)
            log_file_handler = logging.FileHandler(log_file_path)
            log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            test_logger.addHandler(log_file_handler)
        
        # load a trained model
        model = LSTM_Trajectory_Forecast(cfg=cfg.model_params)
        model.load_state_dict(torch.load(f=os.path.join(cfg.checkpoint_path, "model_final.pt"), map_location=device)["model"])
        
        # pretrained model...?
        sequence_len = max_input_horizon
        model.forward(torch.Tensor([[[batch_size], [sequence_len], [lstm_input_shape]]]))
        
        # create eval forecaster
        eval_forecaster = MDN_Forecaster(cfg=cfg, model=model, data_loader=data_loader, type='testing', device=device, logger=test_logger)
        
        if cfg.with_print: print(colored(f"Start testing for: \n - config: {cfg.name} \n - target: {cfg.target} \n - model_arch: {cfg.model_arch} \n - model parameters: {count_model_parameters(model=model)}", 'green'))
        if cfg.with_log: test_logger.info(f"Start testing for: \n - config: {cfg.name} \n - target: {cfg.target} \n - model_arch: {cfg.model_arch} \n - model parameters: {count_model_parameters(model=model)}")
        
        # Run evaluation tasks
        eval_forecaster.evaluate(epoch=None)
        
        # save example plots
        if plot_examples or plot_examples_to_map:
            
            if cfg.with_print: print(colored(f"Plotting {int(len(data_loader.test_data[0])/plot_step)} examples...", 'magenta'))
            if cfg.with_log: test_logger.info(f"Plotting {int(len(data_loader.test_data[0])/plot_step)} examples...")
            
            eval_forecaster.save_examples(
                epoch=None, 
                n_samples=num_samples,
                mesh_range_x=mesh_range_x,
                mesh_range_y=mesh_range_y, 
                mesh_resolution=mesh_resolution, 
                confidence_levels=confidence_levels,
                plot_ego=plot_examples,
                plot_map=plot_examples_to_map
                )
        
        if cfg.with_print: print(colored(f"Finished testing...", 'green'))
        if cfg.with_log: test_logger.info(f"Finished testing...")
    
    if cfg.with_print: print(colored(f"All tests completed, shutdown...", 'green'))
    if cfg.with_log: test_logger.info(f"All tests completed, shutdown...")
    
    if cfg.with_log:
        
        log_file_handler.close()
        test_logger.removeHandler(log_file_handler)
        
    print(colored(f"Finished testing: {cfg.name} on GPU: {gpu_id}", 'cyan'))
        
    return


if __name__ == "__main__":
    
    parser = config_parser()
    args=parser.parse_args()
    
    # gpu handling
    if args.gpu:
        
        gpu_id = args.gpu
        
    else:
        
        gpu_id = 0
        
    testing(args=args, gpu_id=gpu_id)