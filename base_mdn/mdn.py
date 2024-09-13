import os
import torch
import numpy as np
import time

from eval import MDN_Forecaster
from statistics import mean
from vis import plot_train_loss
from termcolor import colored


class MDN_Trainer:
    """MDN trainer class
    """
    
    def __init__(self, cfg, model, loss_fn, optimizer, scheduler, epoch, loss_hist, logger, device='cpu'):
        """Init and setup
        """
        
        if cfg.with_print: print(colored(f"Init trainer...", 'green'))
        if cfg.with_log: logger.info(f"Init trainer...")
        
        # trainer params
        self.cfg = cfg
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epoch = epoch
        self.train_loss_per_epoch_list = loss_hist
        self.logger = logger
        
        # model params
        self.dt = cfg.model_params['delta_t']
        self.num_gaussians = cfg.model_params['num_gaussians']
        self.forecast_horizon = cfg.model_params['forecast_horizon']
        
        # training params
        self.dynamic_input_horizon = cfg.train_params['dynamic_input_horizon']
        self.resume_training = cfg.train_params['resume_training']
        self.train_batch_size = cfg.train_params['batch_size']
        self.train_epochs = cfg.train_params['train_epochs']
        self.eval_epoch_step = cfg.train_params['eval_epoch_step']
        self.min_dynamic_input_horizon = cfg.train_params['min_dynamic_input_horizon']
        
        # eval params
        self.confidence_levels = cfg.test_params['confidence_levels']
        self.mesh_range_x = cfg.test_params['mesh_range_x']
        self.mesh_range_y = cfg.test_params['mesh_range_y']
        self.mesh_resolution = cfg.test_params['mesh_resolution']
        self.num_samples = cfg.test_params['num_samples']
        self.plot_examples = cfg.train_params['plot_examples']
        self.plot_step = cfg.train_params['plot_step']
        self.plot_map_examples = cfg.train_params['plot_examples_to_map']
        self.plot_map_step = cfg.train_params['plot_map_step']
        self.test_horizons = cfg.test_params['test_horizons']
        
        return
    
    
    def save(self, epoch, diverged, final):
        """Save model
        """
        
        # create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.train_loss_per_epoch_list
        }
        
        # save model
        if final:
            
            torch.save(obj=checkpoint, f=os.path.join(self.cfg.checkpoint_path, "model_final.pt"))
            
        else:
            
            torch.save(obj=checkpoint, f=os.path.join(self.cfg.checkpoint_path, f"{self.cfg.name}_{str(epoch).zfill(4)}.pt"))
        
        # training diverged, i.e stop it and save latest status
        if diverged:
            
            torch.save(obj=checkpoint, f=os.path.join(self.cfg.checkpoint_path, "model_final.pt"))
            if self.cfg.with_print: print(colored(f"MDN Loss, training diverged, saving model and shutting down", 'red'))
            if self.cfg.with_log: self.logger.info(f"MDN Loss, training diverged, saving model and shutting down")
            return
        
        # continue
        else:
            
            return
        
        
    def train(self, data_loader):
        """Run training
        """
        
        if self.cfg.with_print: print(colored(f"Start training...", 'green'))
        if self.cfg.with_log: self.logger.info(f"Start training...")
        
        # init
        eval_loss_per_epoch_list = []
        final_epoch = self.epoch
        
        # epochs
        for epoch in range(self.epoch, self.train_epochs+1):
            
            st = time.time()
            train_loss_list = []
            final_epoch += 1
            self.model.train()
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # split train data
            train_data_X, train_data_y = data_loader.get_train_data()
            
            # train one epoch
            for iteration in range(0, len(train_data_X), self.train_batch_size):
                
                inputs = torch.tensor(data=train_data_X[iteration:(iteration+self.train_batch_size)], dtype=torch.float32)
                targets = torch.tensor(data=train_data_y[iteration:(iteration+self.train_batch_size)], dtype=torch.float32)
                
                # with dynamic but consistent input horizon length for every batch
                if self.dynamic_input_horizon:
                    
                    n = np.random.choice(range(self.min_dynamic_input_horizon, inputs.size()[-2], 1))
                    inputs = inputs[:,n:,:]
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss, diverged = self.loss_fn(outputs, targets)
                
                # Check if training diverged
                if diverged: 
                    
                    self.save(epoch=epoch, diverged=diverged, final=False)
                    return final_epoch-1, self.train_loss_per_epoch_list
                    
                # No, continue
                else:
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    # train loss for every batch
                    train_loss_list.append(loss.item())
                
            train_loss = mean(train_loss_list)
            self.scheduler.step()
            
            # process validation data to get validation data loss
            self.model.eval()
            val_loss_list = []
            
            # split eval data
            eval_data_X, eval_data_y, _, _ ,_ = data_loader.get_eval_data()
            
            with torch.no_grad():
                
                for iteration in range(0, len(eval_data_X), self.train_batch_size):
                    
                    inputs = torch.tensor(data=eval_data_X[iteration:(iteration+self.train_batch_size)], dtype=torch.float32)
                    targets = torch.tensor(data=eval_data_y[iteration:(iteration+self.train_batch_size)], dtype=torch.float32)
                    
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    
                    loss, diverged = self.loss_fn(outputs, targets)
                    
                    # Check if model diverged
                    if diverged: 
                        
                        self.save(epoch=epoch, diverged=diverged, final=False)
                        return final_epoch-1, self.train_loss_per_epoch_list
                    
                    # No, continue
                    else:
                        
                        val_loss_list.append(loss.item())
                    
            # epoch completed, save temporal data and update console feedback
            eval_loss = mean(val_loss_list)
            self.train_loss_per_epoch_list.append(train_loss)
            eval_loss_per_epoch_list.append(eval_loss)
            plot_train_loss(self.train_loss_per_epoch_list, self.cfg.evaluation_path, self.cfg.name, self.cfg.model_arch)
            duration = round((time.time() - st), 2)
            eta = round(((duration * (self.train_epochs - epoch)) / 3600), 2)
            if self.cfg.with_print: print(colored(f"Epoch: {epoch}/{self.train_epochs} || eta: {eta} h || time: {duration} secs || Train loss: {train_loss:.4f} || Eval loss: {eval_loss:.4f} || Learning Rate: {current_lr:.8f}", 'yellow'), end='\r')
            if self.cfg.with_log: self.logger.info(f"Epoch: {epoch}/{self.train_epochs} || eta: {eta} h || time: {duration} secs || Train loss: {train_loss:.4f} || Eval loss: {eval_loss:.4f} || Learning Rate: {current_lr:.8f}")
            
            # evaluate reliability and sharpness, therefore pause training and save current model weights
            if (epoch)%self.eval_epoch_step == 0 and epoch != 0:
                
                if self.cfg.with_print: print(colored(f"\nEpoch: {epoch}/{self.train_epochs} || Evaluating...", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"Epoch: {epoch}/{self.train_epochs} || Evaluating...")
                
                # save model
                self.save(epoch=epoch, diverged=False, final=False)
                
                # create mdn eval forecaster
                eval_forecaster = MDN_Forecaster(cfg=self.cfg, model=self.model, data_loader=data_loader, type='eval', device=self.device, logger=self.logger)
                
                # run evaluation tasks
                eval_forecaster.evaluate(epoch=epoch)
                
                # create and save example plots of evaluation samples
                if self.plot_examples or self.plot_map_examples:
                    
                    if self.cfg.with_print: print(colored(f" - plotting {int(len(data_loader.eval_data[0])/self.plot_step)} examples...", 'magenta'))
                    if self.cfg.with_log: self.logger.info(f" - plotting {int(len(data_loader.eval_data[0])/self.plot_step)} examples...")
                    
                    eval_forecaster.save_examples(
                        epoch=epoch, 
                        n_samples=self.num_samples,
                        mesh_range_x=self.mesh_range_x,
                        mesh_range_y=self.mesh_range_y, 
                        mesh_resolution=self.mesh_resolution, 
                        confidence_levels=self.confidence_levels,
                        plot_ego=self.plot_examples,
                        plot_map=self.plot_map_examples
                    )
                
        if self.cfg.with_print: print(colored(f"Finished training, running {final_epoch-1}/{self.train_epochs} epochs", 'green'))
        if self.cfg.with_log: self.logger.info(f"Finished training, running {final_epoch-1}/{self.train_epochs} epochs")
        
        return final_epoch-1, self.train_loss_per_epoch_list