import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import cv2 as cv


class VisIntersectionMap:
    """Class to draw 3D IMPTC tracks to 2D topview map as image
    """
    
    def __init__(self):
        
        # scale parameter
        self.s = 30
        
        # rotation angle
        self.gamma = 0.57 * math.pi
        
        # rotation matrix
        self.R = np.matrix([[math.cos(self.gamma), -math.sin(self.gamma)],
                            [math.sin(self.gamma), math.cos(self.gamma)]])
        
        # translation vector
        self.t = np.matrix([[760], [447]])
        
        # color handling
        self.colors = []
        self.single_colors = []
        self.range_colors = []
        self.n_colors = 1
        return
        
        
    def transform_point(self, p):
        """Transform a 3D point from IMPTC world into 2D topview map
        """
        
        # scale
        p *= self.s
        
        # rotate
        p = self.R * p.transpose()
        
        # translate
        p += self.t
        
        return p
    
    
    def draw_contour(self, topview_map, contours, modes, kappas, colors):
        """Draw certain confidence areas as contours to topview map
        """
        
        for idx, _ in enumerate(kappas):
            
            alpha = (1/(len(kappas)) + idx*(1/(len(kappas)))) * 0.75
            
            for k, contour in reversed(list(enumerate(contours[idx]))):
                
                color = colors[k] * 255
                
                self.draw_point(topview_map=topview_map, x=modes[k,0], y=modes[k,1], radius=6, color=color)
                
                for m in contour:
                    
                    uv = []
                    for n in range(m.shape[0]):
                        
                        x = np.squeeze(m[n,0])
                        y = np.squeeze(m[n,1])
                        
                        p1 = np.matrix([x, -y])
                        p1 = self.transform_point(p1)
                        uv.append([int(p1[0]), int(p1[1])])
                        
                    overlay = topview_map.copy()
                    overlay = cv.drawContours(overlay, contours=np.array(uv)[None,...], contourIdx=-1, color=color, thickness=cv.FILLED)
                    topview_map = cv.addWeighted(topview_map, 1-alpha, overlay, alpha, 0)
        
        return topview_map
    
    
    def draw_point(self, topview_map, x, y, radius, color):
        """Draw a single point into topview map
        """
        
        p1 = np.matrix([x, -y])
        p1 = self.transform_point(p1)
        u, v = (int(p1[0]), int(p1[1]))
        topview_map = cv.circle(topview_map, center=(u, v), color=color, radius=radius, thickness=-1)
        
        return
    
    
    def draw_trajectory_as_line(self, topview_map, track, radius, color):
        """Draw complete trajectory to topview map
        """
        
        uv_l = []
        
        for t in track:
            
            x = t[0]
            y = t[1]
                
            p1 = np.matrix([x, -y])
            p1 = self.transform_point(p1)
            uv_l.append([int(p1[0]), int(p1[1])])
            
        for p1, p2 in zip(uv_l, uv_l[1:]):
            
            cv.line(img=topview_map, pt1=p1, pt2=p2, color=color, thickness=radius) 
            
        return
    
    
    
class VisIndMap:
    """Class to draw 3D inD dataset tracks to 2D topview map as single image
    """
    
    def __init__(self, src):
        
        self.src = int(src)
        self.scale_down = 12
        self.ortho_px_to_meter = 0
        
        # define factor for world to pixel transformation
        self.update_src(src=src)
        
        # color handling
        self.colors = []
        self.single_colors = []
        self.range_colors = []
        self.n_colors = 1
        return
        
        
    def update_src(self, src):
        """Change topview map source and parameters
        """
        
        self.src = int(src)
        
        # define factor for world to pixel transformation
        if self.src < 0:
            
            self.ortho_px_to_meter = 0
            
        elif self.src <= 6:
            
            self.ortho_px_to_meter = 0.0126999352667008
            
        elif self.src >= 7 and self.src <= 17:
            
            self.ortho_px_to_meter = 0.00814636091724916
            
        elif self.src >= 18 and self.src <= 29:
            
            self.ortho_px_to_meter = 0.00814636091724502
            
        elif self.src >= 30 and self.src <= 32:
            
            self.ortho_px_to_meter = 0.00814635379575616
            
        return
        
        
    def transform_point(self, p):
        """ Transform a 3d point from xung into 2d topview map
        Args:
            p (np.array): 3d xung position, only x and y
        Returns:
            np.array: transformed u,v image pixel coordinates
        """
        
        p = p / (self.ortho_px_to_meter * self.scale_down)
        
        return p
    
    
    def draw_contour(self, topview_map, contours, modes, kappas, colors):
        """Draw certain confidence areas as contours to topview map
        """
        
        for idx, _ in enumerate(kappas):
            
            alpha = (1/(len(kappas)) + idx*(1/(len(kappas)))) * 0.95
            
            for k, contour in reversed(list(enumerate(contours[idx]))):
                
                color = colors[k] * 255
                
                self.draw_point(topview_map=topview_map, x=modes[k,0], y=modes[k,1], radius=2, color=color)
                
                for m in contour:
                    
                    uv = []
                    for n in range(m.shape[0]):
                        
                        x = np.squeeze(m[n,0])
                        y = np.squeeze(m[n,1])
                        
                        p1 = np.array([x, y])
                        p1 = self.transform_point(p1)
                        uv.append([int(p1[0]), int(p1[1])])
                        
                    overlay = topview_map.copy()
                    overlay = cv.drawContours(overlay, contours=np.array(uv)[None,...], contourIdx=-1, color=color, thickness=cv.FILLED)
                    topview_map = cv.addWeighted(topview_map, 1-alpha, overlay, alpha, 0)
        
        return topview_map
    
    
    def draw_point(self, topview_map, x, y, radius, color):
        """ Draw a single point into topview map
        """
        
        p1 = np.array([x, y])
        p1 = self.transform_point(p1)
        u, v = (int(p1[0]), int(p1[1]))
        topview_map = cv.circle(topview_map, center=(u, v), color=color, radius=radius, thickness=-1)
        
        return
    
    
    def draw_trajectory_as_line(self, topview_map, track, radius, color):
        """Draw complete trajectory to topview map
        """
        
        uv_l = []
        
        for t in track:
            
            x = t[0]
            y = t[1]
            
            p1 = np.array([x, y])
            p1 = self.transform_point(p1)
            uv_l.append([int(p1[0]), int(p1[1])])
            
        for p1, p2 in zip(uv_l, uv_l[1:]):
            
            cv.line(img=topview_map, pt1=p1, pt2=p2, color=color, thickness=radius) 
            
        return
    
    
def plot_train_loss(train_loss_list, dst_dir, cfg_name, model_arch):
    """Plot train loss
    """
    
    plt.figure(figsize=(16.8,10.5))
    plt.plot(train_loss_list)
    plt.xticks(np.arange(0,len(train_loss_list),250))
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.title(f'Train Loss over Epochs for {model_arch} with {cfg_name} config')
    plt.savefig(os.path.join(dst_dir, 'train_loss.png'))
    plt.close()
    
    return


def plot_reliability_calibration(confidence_sets, dst_dir, epoch, dt, steps, bins):
    """Plot reliability calibration curve
    """
    
    # place/sort values into bins
    # attention!: digitize() returns indexes, with first index starting at 1 not 0
    bin_data = np.digitize(confidence_sets, bins=bins)
    reliability_errors = []
    
    plt.figure()
    plt.plot(bins, bins, 'k--', linewidth=3, label=f"ideal")
    
    for idx in range(0, len(steps)):
        
        # build calibration curve
        # attention!: bincount() returns amount of each bin, first bin to count is bin at 0,
        # due to digitize behavior must increment len(bins) by 1 and later ignore the zero bin count
        f0 = np.array(np.bincount(bin_data[:,idx], minlength=len(bins)+1)).T
        
        # f0[1:]: because of the different start values of digitize and bincount, we remove/ignore the first value of f0
        acc_f0 = np.cumsum(f0[1:],axis=0)/confidence_sets.shape[0]
        
        # get differences for current step
        r = abs(acc_f0 - bins)
        reliability_errors.append(r)
        
        # visualize
        plt.plot(bins,acc_f0, linewidth=3, label=f"{round((steps[idx]+1)*dt, 1)} sec @ avg: {(1 - np.mean(r))*100:.1f} %, min: {(1 - np.max(r))*100:.1f} %")
        
    # get reliability scores
    reliability_avg_score = (1 - np.mean(reliability_errors))*100
    reliability_min_score = (1- np.max(reliability_errors))*100
    
    plt.grid()
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.title(f'Reliability at epoch: {epoch} - Avg: {reliability_avg_score:.1f} % - Min: {reliability_min_score:.1f} %')
    
    # save plot
    path = os.path.join(dst_dir + '/reliability')
    
    if not os.path.exists(path): os.mkdir(path)

    plt.savefig(os.path.join(path, f'reliability_epoch_{str(epoch).zfill(4)}_plain.png'))
    plt.legend(fontsize = 10)
    plt.savefig(os.path.join(path, f'reliability_epoch_{str(epoch).zfill(4)}_legend.png'))
        
    plt.close()
    return reliability_avg_score, reliability_min_score, reliability_errors


def plot_sharpness_over_time(data, dst_dir, epoch, dt, confidence_levels, steps, num_steps):
    """Plot sharpness areas over time
    """
    
    # data shape: [n_samples, confidence_levels, n_horizons]
    colors = sns.color_palette("hls", 8)
    n_horizons = data.shape[-1]
    
    # define percentiles
    percentiles = [k for k in np.arange(0.0, 1.01, 0.01)]
    
    # build axis in s
    x = [round((k+1) * dt, 1) for k in steps]
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(16.8,10.5))
    
    sharpness_scores_list = []
    
    for idx, kappa in enumerate(confidence_levels):
        
        sample_data = data[:,idx,:]
        sample_data = sample_data.T
        n_horizons = sample_data.shape[0]
        SDist=np.zeros((n_horizons, len(percentiles)))
        
        for k, p in enumerate(percentiles):
            
            for t in range(n_horizons):
                
                SDist[t,k]=np.percentile(sample_data[t,:], p*100, axis=-1)
        
        # calc mean sharpness score (i.e 50%) for current kappa
        sharpness_score = sum([np.mean(SDist[idx,:] / ((step+1)*dt)) for idx, step in enumerate(steps)]) * (1/(num_steps*dt))
        
        # plot mean (i.e at 50 %) and range from 25 % to 75 %
        ax1.plot(x, SDist[:,int(len(percentiles)/2)], color=colors[idx], label=f"{kappa*100} %, SS: {sharpness_score:.2f} m²/s")
        ax1.fill_between(x, SDist[:,int(len(percentiles)/4)], SDist[:,int(len(percentiles)*3/4)], color=colors[idx], alpha=0.5)
        
        # plot
        #ax1.plot(x, sharpness_score*np.ones_like(x), color=colors[idx], label=f"SS @ {kappa}")
        sharpness_scores_list.append(sharpness_score)
            
    ax1.set_title("Sharpness Percentiles:", fontsize=15)
    ax1.tick_params(labelsize=11.5)
    ax1.set_xlabel('Forecast Horizon in s', fontsize=14)
    ax1.set_ylabel('Sharpness in m²', fontsize=14)
    plt.xlim([0,4.8])
    plt.ylim([0,4])
    plt.legend(fontsize=16)
    fig.tight_layout()
    
    # save plot
    path = os.path.join(dst_dir, 'sharpness')
    
    if not os.path.exists(path): os.mkdir(path)
        
    if epoch is None: plt.savefig(os.path.join(path, f'sharpness.png'))
        
    else: plt.savefig(os.path.join(path, f"sharpness_epoch_{str(epoch).zfill(4)}.png"))
        
    plt.close()
    return sharpness_scores_list
    
    
def plot_aee_over_time(data, dst_dir, epoch, dt, steps, num_steps):
    """Plot ASAEE/AEE over time
    """
    
    # Data shape: [n_samples, n_horizons]
    n_horizons = data.shape[1]
    
    # define percentiles
    percentiles = [k for k in np.arange(0.0, 1.01, 0.01)]
    
    # build axis in s
    x = [round((k+1) * dt, 1) for k in steps]
    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(16.8,10.5))
    sample_data=data.T
    SDist=np.zeros((n_horizons, len(percentiles)))
    
    ADE = np.median(sample_data)
    FDE = np.median(sample_data[-1])
    ASAEE = [sum((np.mean(sample_data[i]) / ((steps[i]+1)*dt)) for i in range(0, len(sample_data))) / num_steps][0] * (1/dt)
    
    for k, p in enumerate(percentiles):
        
        for t in range(n_horizons):
            
            SDist[t,k]=np.percentile(sample_data[t,:], p*100, axis=-1)
            
    # plot mean AEE
    ax1.plot(x, SDist[:,int(len(percentiles)/2)], color="gray", label=f"AEE")
    ax1.plot(x, ASAEE*np.ones_like(x), 'g--', label=f"ASAEE")
    ax1.plot(x, ADE*np.ones_like(x), 'r--', label=f"ADE")
    ax1.plot(x, FDE*np.ones_like(x), 'b--', label=f"FDE")
    ax1.fill_between(x, SDist[:,int(len(percentiles)/4)], SDist[:,int(len(percentiles)*3/4)], color="gray", alpha=0.5)
        
    ax1.set_title("AEE and ASAEE", fontsize=15)
    ax1.tick_params(labelsize=11.5)
    ax1.set_xlabel('Forecast Horizon in s', fontsize=14)
    ax1.set_ylabel('AEEs in m, and ASAEE in m/s', fontsize=14)
    plt.ylim([0,2])
    plt.legend()
    fig.tight_layout()
    
    # save plot
    path = os.path.join(dst_dir, 'aee')
    
    if not os.path.exists(path):
        
        os.mkdir(path)
        
    if epoch is None:
        
        plt.savefig(os.path.join(path, f'aee.png'))
        
    else:
    
        plt.savefig(os.path.join(path, f"aee_epoch_{str(epoch).zfill(4)}.png"))
        
    plt.close()
    return ASAEE


def plot_ego_forecast(cfg, X, y, forecasts, modes, dst_dir, sample_id, epoch, confidence_levels, ade, fde, src):
    """Plot ego forecast
    """
    
    # dst dir and epoch handling
    if epoch == None:
        
        dst_epoch_dir = dst_dir
        
    else:
        
        dst_epoch_dir = os.path.join(dst_dir, 'epoch_' + str(epoch).zfill(4))
        
    if not os.path.exists(dst_epoch_dir):
        
        os.makedirs(dst_epoch_dir)
        
    gt_data = []
    mode_data = []
    for s in range(0, len(cfg.test_params['test_horizons'])):
        
        gt_data.append(y[:,s,:])
        mode_data.append(modes[s,:])
    
    # create figure
    fig, ax = plt.subplots(figsize=(16.8,10.5))
    
    # plot most likely positions
    for l, m in enumerate(mode_data):
        
        color = cfg.colors_rgb[l]
        label = 'Most likely @ ' + str(round((cfg.test_params['test_horizons'][l] + 1) * cfg.model_params['delta_t'], 1)) + ' s'
        plt.plot(m[0], m[1], marker='.', markersize=20, color=color, label=label)
    
    # plot confidence areas
    for idx, kappa in enumerate(confidence_levels):
        
        for k, contours in enumerate(forecasts[idx]):
            
            color = cfg.colors_rgb[k]
            
            for n, c in enumerate(contours):
                
                polygon = plt.Polygon(c, facecolor=color, edgecolor=color, alpha=0.25+(idx*0.25))
                ax.add_patch(polygon)
                    
    # plot input and gt data
    plt.plot(np.squeeze(X[:,:,0]), np.squeeze(X[:,:,1]), color='r', linewidth=2, label="Inputs", marker='.', markersize=16)
    plt.plot(np.array(gt_data)[...,0], np.array(gt_data)[...,1], color='k', linewidth=2, label="Ground Truth", marker='.', markersize=16)
    
    plt.gca().set_aspect('equal')
    plt.grid()
    plt.title(f'Sample: {sample_id} || ADE: {ade} || FDE: {fde} || Source: {src}')
    plt.xlabel("x / m", fontsize = 24)
    plt.ylabel("y / m", fontsize = 24)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.xlim([-12, 12])
    plt.ylim([-8, 16])
    plt.legend(loc='upper right', fontsize=16)
    plt.savefig(os.path.join(dst_epoch_dir, f'sample_{str(sample_id).zfill(8)}.png'))
    plt.close()
    
    return