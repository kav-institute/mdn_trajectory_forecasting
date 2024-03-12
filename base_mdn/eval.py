import torch
import time
import os
import copy
import cv2 as cv
import numpy as np
import torch.distributions as dist

from skimage import measure
from tqdm import trange
from termcolor import colored
from vis import plot_reliability_calibration, plot_aee_over_time, plot_sharpness_over_time, plot_ego_forecast
from vis import VisIntersectionMap, VisIndMap
from utils.helper import ego2world


class MDN_Forecaster:
    """Inference and evaluation for mdn
    """
    
    def __init__(self, cfg, model, data_loader, type, logger, device='cpu'):
        """Init and setup
        """
        
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.type = type
        self.logger = logger
        self.train_params = cfg.train_params
        self.test_params = cfg.test_params
        self.model_params = cfg.model_params
        self.eval_metrics = cfg.eval_metrics
        self.num_gaussians = self.model_params['num_gaussians']
        self.dt = self.model_params['delta_t']
        self.forecast_horizon = self.model_params['forecast_horizon']
        self.batch_size = self.test_params['batch_size']
        self.n_samples = self.test_params['num_samples']
        self.eval_input_horizons = self.test_params['num_input_horizons']
        self.reliability_bins = cfg.reliability_bins
        
        # metrics
        self.with_ade_fde = self.eval_metrics['ade_fde']
        self.with_ade_fde_k = self.eval_metrics['ade_fde_k']
        self.with_reliability = self.eval_metrics['reliability']
        self.with_sharpness = self.eval_metrics['sharpness']
        self.with_asaee = self.eval_metrics['asaee']
        
        # forecasting use case
        # eval while training
        if type == 'eval':
            
            self.plot_examples = self.train_params['plot_examples']
            self.plot_step = self.train_params['plot_step']
            self.plot_ego_dst_dir = cfg.eval_ego_examples_path
            self.plot_world_dst_dir = cfg.eval_world_examples_path
            self.data_X, self.data_y, self.data_ref, self.data_rot, self.data_src = data_loader.get_eval_data()
            self.dst_dir = cfg.evaluation_path
        
        # testing a trained model with test data
        elif type == 'testing':
            
            self.plot_examples = self.test_params['plot_examples']
            self.plot_step = self.test_params['plot_step']
            self.plot_ego_dst_dir = cfg.test_ego_examples_path
            self.plot_world_dst_dir = cfg.test_world_examples_path
            self.data_X, self.data_y, self.data_ref, self.data_rot, self.data_src = data_loader.get_test_data()
            self.dst_dir = cfg.testing_path
            
        return
        
        
    def evaluate(self, epoch=None):
        """Evaluate model
        """
        
        self.model.eval()
        confidence_sets = []
        sharpness_sets = []
        aee_sets = []
        k_samples_aee_sets = []
        euc_errors_sets = []
        batch_run_time = []
        
        # create meash grid for ego coordinates
        grid = self.build_mesh_grid(mesh_range_x=self.test_params['mesh_range_x'], mesh_range_y=self.test_params['mesh_range_y'], mesh_resolution=self.test_params['mesh_resolution'])
        
        with torch.no_grad():
            
            # batch processing
            for iteration in trange(0, len(self.data_X), self.batch_size):
                
                sharpness_batch = []
                modes_batch = []
                
                # upload data and run model
                inputs = torch.tensor(data=self.data_X[iteration:(iteration+self.batch_size)][:,-self.eval_input_horizons:,:], dtype=torch.float32).to(self.device)
                targets = torch.tensor(data=self.data_y[iteration:(iteration+self.batch_size)], dtype=torch.float32).to(self.device)
                
                st = time.time()
                outputs = self.model(inputs)
                et = time.time()
                
                # measure batch model inference time
                batch_run_time.append((et-st)*1000)
                
                # only use user defined output steps
                filtered_outputs = torch.stack([outputs[:,k,:] for k in self.test_params['test_horizons']], dim=1)
                filtered_targets = torch.stack([targets[:,k,:] for k in self.test_params['test_horizons']], dim=1)
                
                if self.with_ade_fde_k:
                    
                    # get k samples from distributions
                    num_k_samples = self.sample_mdn(filtered_outputs, num_gaussians=self.num_gaussians, n_samples=self.test_params['num_k_samples'])
                    
                    # get euclidean errors for k samples over defined future time steps of batch
                    euc_errors = torch.sqrt(torch.square(filtered_targets - num_k_samples).sum(dim=-1)).cpu().numpy()
                    euc_errors = np.transpose(euc_errors, (1, 2, 0))
                    
                    # get smallest errors of k samples over defined future time steps of batch
                    k_samples_aee_sets.append(euc_errors.min(axis=2))
                    
                    # get smallest error of k trajectories over defined future time steps of batch
                    euc_errors_sets.append(euc_errors)
                
                if self.with_reliability:
                    
                    # build confidence sets
                    confidence_sets.append(self.build_confidence_set_mdn(output=filtered_outputs, target=filtered_targets, num_gaussians=self.num_gaussians, n_samples=self.n_samples).cpu().numpy())
                    
                if self.with_sharpness or self.with_ade_fde or self.with_asaee:
                    
                    # do for every single sample one by one
                    for idx in range(0, filtered_outputs.shape[0]):
                    
                        # build confidence map for every sample
                        conf_map = self.build_confidence_set_mdn(output=filtered_outputs[idx][None,...] , target=grid, num_gaussians=self.num_gaussians, n_samples=self.n_samples)
                        
                        if self.with_sharpness:
                            
                            # handle different confidence_levels
                            sharpness = []
                            for k in self.test_params['confidence_levels']:
                                
                                # calc sharpness
                                sharpness.append(self.estimate_sharpness(conf_map, kappa=k)*(self.test_params['mesh_range_x']*self.test_params['mesh_range_y']))
                                
                            sharpness_batch.append(torch.stack(sharpness, 0))
                            
                        if self.with_asaee or self.with_ade_fde:
                            
                            # get most likely position for every defined time step for EE calculation
                            modes_batch.append(grid[torch.argmin(conf_map, dim=0, keepdim=True)][0,:,0,:])
                        
                    if self.with_sharpness:
                        
                        # stack sharpness batches
                        sharpness_sets.append(torch.stack(sharpness_batch, 0).cpu().numpy())
                        
                    if self.with_asaee or self.with_ade_fde:
                        
                        # get euclidean errors
                        aee_sets.append(torch.sqrt(torch.square(filtered_targets - torch.stack(modes_batch, 0)).sum(dim=-1)).cpu().numpy())
            
            # average model inference time
            if self.cfg.with_print: print(colored(f"====================", 'magenta'))
            if self.cfg.with_print: print(colored(f" Avg model inference time: {str(round(np.mean(batch_run_time), 2))} ms", 'magenta'))
            if self.cfg.with_log: self.logger.info(f"====================")
            if self.cfg.with_log: self.logger.info(f" Avg model inference time: {str(round(np.mean(batch_run_time), 2))} ms")
            
            if self.with_reliability:
                
                # create reliability plot
                mean_RLS, min_RLS, RLS_bins = plot_reliability_calibration(confidence_sets=np.vstack(confidence_sets), bins=self.reliability_bins, dst_dir=self.dst_dir, epoch=epoch, dt=self.dt, steps=self.test_params['test_horizons'])
                
                if self.cfg.with_print: print(colored(f"====================", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"====================")
                if self.cfg.with_print: print(colored(f"Reliability Scores:", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"Reliability Scores: ")
                
                if self.cfg.with_print: 
                    for idx, step in enumerate(self.test_params['test_horizons']): print(colored(f" avg. RLS @ {round((step+1)*self.dt, 1)} sec: {(1 - np.mean(RLS_bins[idx]))*100:.2f} %", 'magenta'))
                if self.cfg.with_log:
                    for idx, step in enumerate(self.test_params['test_horizons']): self.logger.info(f" avg. RLS @ {round((step+1)*self.dt, 1)} sec: {(1 - np.mean(RLS_bins[idx]))*100:.2f} %")
                    
                if self.cfg.with_print: print(colored(f"--------------------", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"--------------------")
                if self.cfg.with_print: print(colored(f" RLS: avg: {mean_RLS:.2f} %, min: {min_RLS:.2f} %", 'magenta'))
                if self.cfg.with_log: self.logger.info(f" RLS: avg: {mean_RLS:.2f} %, min: {min_RLS:.2f} %")
                
                f = open(os.path.join(self.dst_dir, 'rls.txt'), "w")
                f.write("Reliability Scores:")
                f.write(f"\n--------------------:")
                for idx, step in enumerate(self.test_params['test_horizons']): f.write(f"\n avg. RLS @ {round((step+1)*self.dt, 1)} sec: {(1 - np.mean(RLS_bins[idx]))*100:.2f} %")
                f.write(f"\n--------------------:")
                f.write(f"\n RLS: avg: {mean_RLS:.2f} %, min: {min_RLS:.2f} %")
                f.close()
                
            if self.with_sharpness:
                
                # create sharpness plot
                SS = plot_sharpness_over_time(data=np.vstack(sharpness_sets), dst_dir=self.dst_dir, epoch=epoch, dt=self.dt, confidence_levels=self.test_params['confidence_levels'], steps=self.test_params['test_horizons'], num_steps=self.forecast_horizon)
                
                if self.cfg.with_print: print(colored(f"====================", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"====================")
                if self.cfg.with_print: print(colored(f"Sharpness Score:", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"Sharpness Score: ")
                
                if self.cfg.with_print: 
                    for idx, kappa in enumerate(self.test_params['confidence_levels']): print(colored(f" SS @ {kappa} %: {SS[idx]:.2f} m²/s", 'magenta'))
                if self.cfg.with_log:
                    for idx, kappa in enumerate(self.test_params['confidence_levels']): self.logger.info(f" SS @ {kappa} %: {SS[idx]:.2f} m²/s")
                    
                f = open(os.path.join(self.dst_dir, 'ss.txt'), "w")
                f.write("Sharpness Score:")
                f.write(f"\n--------------------:")
                for idx, kappa in enumerate(self.test_params['confidence_levels']): f.write(f"\n SS @ {kappa} %: {SS[idx]:.2f} m²/s")
                f.close()
                
            if self.with_asaee:
                
                # create aee plot
                ASAEE = plot_aee_over_time(data=np.vstack(aee_sets), dst_dir=self.dst_dir, epoch=epoch, dt=self.dt, steps=self.test_params['test_horizons'], num_steps=self.forecast_horizon)
                
                if self.cfg.with_print: print(colored(f"====================", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"====================")
                if self.cfg.with_print: print(colored(f"ASAEE:", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"ASAEE: ")
                
                if self.cfg.with_print: print(colored(f" ASAEE: {ASAEE:.2f} m/s", 'magenta'))
                if self.cfg.with_log: self.logger.info(f" ASAEE: {ASAEE:.2f} m/s")
                
                f = open(os.path.join(self.dst_dir, 'asaee.txt'), "w")
                f.write(f"ASAEE: {ASAEE:.2f} m/s")
                f.close()
                
            if self.with_ade_fde:
                
                ade_by_steps = [round(np.mean(n),3) for n in np.vstack(aee_sets).T]
                ade_final = np.mean(ade_by_steps)
                
                if self.cfg.with_print: print(colored(f"====================", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"====================")
                if self.cfg.with_print: print(colored(f"Most likely: ", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"Most likely: ")
                
                if self.cfg.with_print: 
                    for idx, step in enumerate(self.test_params['test_horizons']): print(colored(f" EE @ {round((step+1)*self.dt, 1)} sec: {str(round(ade_by_steps[idx], 2))} m", 'magenta'))
                if self.cfg.with_log:
                    for idx, step in enumerate(self.test_params['test_horizons']): self.logger.info(f" EE @ {round((step+1)*self.dt, 1)} sec: {str(round(ade_by_steps[idx], 2))} m")
                    
                if self.cfg.with_print: print(colored(f"--------------------", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"--------------------")
                if self.cfg.with_print: print(colored(f" ADE: {str(round(ade_final, 2))} m", 'magenta'))
                if self.cfg.with_log: self.logger.info(f" ADE: {str(round(ade_final, 2))} m")
                
                fde_final = [round(np.mean(n),3) for n in np.vstack(aee_sets).T][-1]
                if self.cfg.with_print: print(colored(f" FDE: {str(round(fde_final, 2))} m", 'magenta'))
                if self.cfg.with_log: self.logger.info(f" FDE: {str(round(fde_final, 2))} m")
                
                f = open(os.path.join(self.dst_dir, 'ade_fde.txt'), "w")
                f.write("Most likely:")
                f.write(f"\n--------------------:")
                for idx, step in enumerate(self.test_params['test_horizons']): f.write(f"\n EE @ {round((step+1)*self.dt, 1)} sec: {str(round(ade_by_steps[idx], 2))} m")
                f.write(f"\n--------------------:")
                f.write(f"\nADE: {str(round(ade_final, 2))} m \nFDE: {str(round(fde_final, 2))} m")
                f.close()
                
            if self.with_ade_fde_k:
                
                # ade of best of k samples
                k_ade_by_steps = [round(np.mean(n),3) for n in np.vstack(k_samples_aee_sets).T]
                k_ade_final = np.mean(k_ade_by_steps)
                if self.cfg.with_print: print(colored(f"====================", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"====================")
                if self.cfg.with_print: print(colored(f"Best of K ({self.test_params['num_k_samples']}) samples for each future horizon: ", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"Best of K ({self.test_params['num_k_samples']}) samples for each future horizon: ")
                
                if self.cfg.with_print: 
                    for idx, step in enumerate(self.test_params['test_horizons']): print(colored(f" EE @ {round((step+1)*self.dt, 1)} sec: {str(round(k_ade_by_steps[idx], 2))} m", 'magenta'))
                if self.cfg.with_log: 
                    for idx, step in enumerate(self.test_params['test_horizons']): self.logger.info(f" EE @ {round((step+1)*self.dt, 1)} sec: {str(round(k_ade_by_steps[idx], 2))} m")
                
                if self.cfg.with_print: print(colored(f"--------------------", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"--------------------")
                if self.cfg.with_print: print(colored(f" ADE: {str(round(k_ade_final, 2))} m", 'magenta'))
                if self.cfg.with_log: self.logger.info(f" ADE: {str(round(k_ade_final, 2))} m")
                
                # fde of best of k samples
                k_fde_final = [round(np.mean(n),3) for n in np.vstack(k_samples_aee_sets).T][-1]
                if self.cfg.with_print: print(colored(f" FDE: {str(round(k_fde_final, 2))} m", 'magenta'))
                if self.cfg.with_log: self.logger.info(f" FDE: {str(round(k_fde_final, 2))} m")
                
                # # ade of best of k trajectories
                k_euc_errors = np.vstack(euc_errors_sets)
                min_ids = np.argmin(np.mean(k_euc_errors, axis=1), axis=1)
                k_ade_trajectories_by_step = np.mean(np.array([k_euc_errors[n,:,i] for n, i in enumerate(min_ids)]), axis=0)
                k_ade_trajectories_final = np.mean(np.min(np.mean(k_euc_errors, axis=1), axis=1))
                
                if self.cfg.with_print: print(colored(f"====================", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"====================")
                if self.cfg.with_print: print(colored(f"Best of K ({self.test_params['num_k_samples']}) full sampled trajectories: ", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"Best of K ({self.test_params['num_k_samples']}) full sampled trajectories: ")
                
                if self.cfg.with_print: 
                    for idx, step in enumerate(self.test_params['test_horizons']): print(colored(f" EE @ {round((step+1)*self.dt, 1)} sec: {str(round(k_ade_trajectories_by_step[idx], 2))} m", 'magenta'))
                if self.cfg.with_log: 
                    for idx, step in enumerate(self.test_params['test_horizons']): self.logger.info(f" EE @ {round((step+1)*self.dt, 1)} sec: {str(round(k_ade_trajectories_by_step[idx], 2))} m")
                
                if self.cfg.with_print: print(colored(f"--------------------", 'magenta'))
                if self.cfg.with_log: self.logger.info(f"--------------------")
                if self.cfg.with_print: print(colored(f" ADE: {str(round(k_ade_trajectories_final, 2))} m", 'magenta'))
                if self.cfg.with_log: self.logger.info(f" ADE: {str(round(k_ade_trajectories_final, 2))} m")
                
                # fde of best of k trajectories
                if self.cfg.with_print: print(colored(f" FDE: {str(round(k_ade_trajectories_by_step[-1], 2))} m", 'magenta'))
                if self.cfg.with_log: self.logger.info(f" FDE: {str(round(k_ade_trajectories_by_step[-1], 2))} m")
                
                f = open(os.path.join(self.dst_dir, 'ade_fde_k.txt'), "w")
                f.write(f"Best of K ({self.test_params['num_k_samples']}):")
                f.write(f"\n--------------------:")
                for idx, step in enumerate(self.test_params['test_horizons']): f.write(f"\n EE @ {round((step+1)*self.dt, 1)} sec: {str(round(k_ade_trajectories_by_step[idx], 2))} m")
                f.write(f"\n--------------------:")
                f.write(f"\nADE: {str(round(k_ade_trajectories_final, 2))} m \nFDE: {str(round(k_ade_trajectories_by_step[-1], 2))} m")
                f.close()
        
        return
        
        
    def save_examples(self, mesh_range_x, mesh_range_y, mesh_resolution, confidence_levels, n_samples, plot_ego, plot_map, epoch=None):
        """Plot examples
        """
        
        # build_ grid
        steps = int(((mesh_range_x + mesh_range_y) / mesh_resolution) + 1)
        grid = self.build_mesh_grid(mesh_range_x=mesh_range_x, mesh_range_y=mesh_range_y, mesh_resolution=mesh_resolution)
        self.model.eval()
        
        # only for IMPTC and inD datasets
        if plot_map:
            
            if self.cfg.target == 'imptc':
                
                map_base_path = '/workspace/repos/data/imptc'
                map_visualizer = VisIntersectionMap()
            
            elif self.cfg.target == 'ind':
                
                map_base_path = '/workspace/repos/data/ind'
                map_visualizer = VisIndMap(src='-1')
        
        with torch.no_grad():
            
            for p in range(0, len(self.data_X), self.plot_step):
                
                conf_areas = []
                conf_contours_ego = []
                conf_contours_world = []
                
                input = torch.tensor(self.data_X[p:p+1][:,-self.eval_input_horizons:,:], dtype=torch.float32).to(self.device)
                target = torch.tensor(self.data_y[p:p+1], dtype=torch.float32).to(self.device)
                reference_pos = self.data_ref[p]
                rotation_angle = self.data_rot[p]
                src = self.data_src[p]
                
                # run model
                output = self.model(input)
                
                # only use user defined output steps
                filtered_output = torch.stack([output[:,k,:] for k in self.test_params['test_horizons']], dim=1)
                filtered_target = torch.stack([target[:,k,:] for k in self.test_params['test_horizons']], dim=1)
                
                # build confidence sets
                conf_map = self.build_confidence_set_mdn(output=filtered_output, target=grid, num_gaussians=self.num_gaussians, n_samples=n_samples)
                modes = grid[torch.argmin(conf_map, dim=0, keepdim=True)]
                modes = modes[0,:,0,:]
                
                # get ade and fde
                EE = torch.sqrt(torch.square(filtered_target-modes).sum(dim=-1)).cpu().numpy()
                ADE = str(round(np.mean(EE.T),3))
                FDE = str(round(np.mean(EE.T[-1]),3))
                
                for k in confidence_levels:
                    
                    conf_areas.append(torch.reshape(torch.where(conf_map <= k, 1, 0), (steps, steps, filtered_output.shape[1])).cpu().numpy()) 
                
                for idx, _ in enumerate(confidence_levels):
                    
                    contours_ego = []
                    contours_world = []
                    
                    for h in range(0, len(self.cfg.test_params['test_horizons'])):
                        
                        # get contour(s) of confidence level and time step
                        c = measure.find_contours(conf_areas[idx][:, :, h], 0.5)
                        
                        # uni modal dist
                        if len(c) == 1:
                            
                            cont_ego = [np.flip(m=np.squeeze(np.array(c, dtype=np.float32) * mesh_resolution - mesh_range_y))]
                            cont_world = ego2world(X=cont_ego, rotation_angle=rotation_angle, translation=reference_pos)
                            contours_ego.append([np.squeeze(a=cont_ego, axis=0)])
                            contours_world.append([np.squeeze(a=cont_world, axis=0)])
                            
                        # multi modal dist
                        elif len(c) >= 2:
                            
                            cont_ego = [np.array(c[i], dtype=np.float32) * mesh_resolution - mesh_range_y for i in range(0, len(c))]
                            cont_ego = [np.flip(m=ct) for ct in cont_ego]
                            cont_world = [ego2world(X=ct, rotation_angle=rotation_angle, translation=reference_pos) for ct in cont_ego]
                            
                            contours_ego.append(cont_ego)
                            contours_world.append(cont_world)
                            
                        # dist area smaller or equal to single point or grid size resolution, i.e mode of this dist
                        else:
                            
                            cont_ego = [np.array(modes.cpu().numpy()[h], dtype=np.float32)[None, ...]]
                            cont_world = ego2world(X=cont_ego, rotation_angle=rotation_angle, translation=reference_pos)
                            
                            contours_ego.append([np.squeeze(a=cont_ego, axis=0)])
                            contours_world.append([np.squeeze(a=cont_world, axis=0)])
                        
                    conf_contours_ego.append(contours_ego)
                    conf_contours_world.append(contours_world)
                
                # plot forecast example results in ego coordinates
                if plot_ego:
                    
                    plot_ego_forecast(cfg=self.cfg, 
                                        X=input.cpu().numpy(), 
                                        y=filtered_target.cpu().numpy(), 
                                        forecasts=conf_contours_ego, 
                                        modes=modes.cpu().numpy(), 
                                        dst_dir=self.plot_ego_dst_dir, 
                                        sample_id=p, 
                                        epoch=epoch, 
                                        confidence_levels=confidence_levels,
                                        ade=ADE, 
                                        fde=FDE)
                
                # only for IMPTC and inD datasets
                # plot forecast example results in world coordinates
                if plot_map:
                    
                    if self.cfg.target == 'imptc':
                    
                        topview_map_base = cv.imread(os.path.join(map_base_path, 'intersection_map.png'))
                        r_line_in = 6
                        r_line_out = 4
                        r_point = 10
                        
                    elif self.cfg.target == 'ind':
                        
                        s = src.split('_')[0]
                        map_visualizer.update_src(src=s)
                        topview_map_base = cv.imread(os.path.join(map_base_path, s + '_background.png'))
                        r_line_in = 2
                        r_line_out = 2
                        r_point = 3
                    
                    input_ego = np.squeeze(input.cpu().numpy(), axis=0)[:,:2]
                    target_ego = np.squeeze(target.cpu().numpy(), axis=0)[:,:2]
                    modes_ego = modes.cpu().numpy()
                    
                    input_world = ego2world(X=input_ego, rotation_angle=rotation_angle, translation=reference_pos)
                    target_world = ego2world(X=target_ego, rotation_angle=rotation_angle, translation=reference_pos)
                    modes_world = ego2world(X=modes_ego, rotation_angle=rotation_angle, translation=reference_pos)
                    
                    top_view_map = copy.deepcopy(topview_map_base)
                    top_view_map = map_visualizer.draw_contour(topview_map=top_view_map, contours=conf_contours_world, modes=modes_world, kappas=confidence_levels, colors=self.cfg.colors_bgr)
                    map_visualizer.draw_trajectory_as_line(topview_map=top_view_map, track=input_world, radius=r_line_in, color=(0,0,255))
                    map_visualizer.draw_trajectory_as_line(topview_map=top_view_map, track=target_world, radius=r_line_out, color=(0,0,0))
                    map_visualizer.draw_point(topview_map=top_view_map, x=input_world[-1,0], y=input_world[-1,1], radius=r_point, color=(0,0,255))
                    
                    if epoch:
                        
                        d = os.path.join(self.plot_world_dst_dir, 'epoch_' + str(epoch).zfill(4))
                        if not os.path.exists(d): os.makedirs(d)
                        map_name = os.path.join(d, f"sample_{str(p).zfill(8)}.png")
                        
                    else:
                        
                        map_name = os.path.join(self.plot_world_dst_dir, f"sample_{str(p).zfill(8)}.png")
                        
                    cv.imwrite(map_name, top_view_map)
                
        return
    
    
    def build_mesh_grid(self, mesh_range_x, mesh_range_y, mesh_resolution):
        
        # build grid
        steps = int(((mesh_range_x + mesh_range_y) / mesh_resolution) + 1)
        xs = torch.linspace(-mesh_range_x, mesh_range_x, steps=steps)
        ys = torch.linspace(-mesh_range_y, mesh_range_y, steps=steps)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        grid = torch.stack([x, y],dim=-1)
        grid = grid.reshape((-1,2))[:,None,:]
        grid = grid.to(self.device)
        return grid
    
    
    def build_distribution(self, output, num_gaussians):
        
        # split the output into the parameters for each Gaussian
        mu_x = output[..., :num_gaussians]
        mu_y = output[..., num_gaussians:2*num_gaussians]
        sigma_x = torch.exp(output[..., 2*num_gaussians:3*num_gaussians])
        sigma_y = torch.exp(output[..., 3*num_gaussians:4*num_gaussians])
        rho = torch.tanh(output[..., 4*num_gaussians:5*num_gaussians])
        alpha = torch.softmax(output[..., 5*num_gaussians:], dim=1)
        
        covs = 0*torch.stack(2*[torch.stack([mu_x, mu_y], dim=-1)],dim=-1).to(output.device)
        covs[..., 0, 0] = sigma_x ** 2
        covs[..., 0, 1] = rho * sigma_x * sigma_y
        covs[..., 1, 0] = rho * sigma_x * sigma_y
        covs[..., 1, 1] = sigma_y ** 2
        
        normal = dist.MultivariateNormal(torch.stack([mu_x, mu_y], dim=-1), covs)
        mixture = dist.Categorical(alpha)
        gmm = dist.MixtureSameFamily(mixture, normal)
        return gmm
        
        
    def build_confidence_set_mdn(self, output, target, num_gaussians, n_samples):
        
        # output shape: [batch_size, n_horizons, num_gaussians * 6] (mu_x, mu_y, sigma_x, sigma_y, rho, alpha for each Gaussian)
        # target shape: [batch_size, n_horizons, 2] (x, y)
        
        # build distribution model
        gmm = self.build_distribution(output=output, num_gaussians=num_gaussians)
        
        gt_log_prob = gmm.log_prob(target)
        samples = gmm.sample(sample_shape=torch.Size([n_samples]))
        samples_log_prob = gmm.log_prob(samples)
        idx_mask = (samples_log_prob > gt_log_prob).float()
        conf = torch.sum(idx_mask, 0)/samples.shape[0]
        return conf
    
    
    def estimate_sharpness(self, confidence_map, kappa):
        
        # confidence_map shape: [n_grid_points, n_horizons]
        # return: confidence area for each horizon [n_horizon]
        area = torch.where(confidence_map <= kappa, 1.0, 0.0)
        area = area.mean(dim=0)
        return area
    
    
    def sample_mdn(self, output, num_gaussians, n_samples):
        
        # build distribution model
        gmm = self.build_distribution(output=output, num_gaussians=num_gaussians)
        
        # create n samples
        samples = gmm.sample(sample_shape=torch.Size([n_samples]))
        return samples