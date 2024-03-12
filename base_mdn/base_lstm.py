import torch
import torch.nn as nn
import torch.distributions as dist


class LSTM_Trajectory_Forecast(nn.Module):
    """LSTM module class
    """
    
    def __init__(self, cfg):
        '''
        The forward method takes in an input tensor x of shape [train_batch_size, sequence_length, lstm_input_shape].
        We pass this input through the LSTM layer using self.lstm(x),
        which returns an output tensor of shape [train_batch_size, sequence_length, lstm_hidden_size].
        We then select the last output of the LSTM sequence (lstm_out[:, -1, :])
        and pass it through the output layer
        (self.fc) to get the final output tensor of shape [train_batch_size, output_size].
        '''
        super(LSTM_Trajectory_Forecast, self).__init__()
        
        self.lstm_input_shape = cfg['lstm_input_shape']
        self.lstm_hidden_size = cfg['lstm_hidden_size']
        self.output_size = cfg['num_gaussians']*cfg['output_factor']
        self.forecast_horizon = cfg['forecast_horizon']
        self.lstm_num_layers = cfg['lstm_num_layers']
        
        # stacked LSTM layer
        self.lstm = nn.LSTM(input_size=self.lstm_input_shape, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_num_layers, batch_first=True)
        
        # output layer
        self.fc = nn.Linear(in_features=self.lstm_hidden_size, out_features=self.output_size*self.forecast_horizon)
        
        
    def forward(self, x):
        """Forward path
        """
        
        # lSTM layer
        # x shape: [train_batch_size, sequence_length, lstm_input_shape]
        # lstm_out shape: [train_batch_size, sequence_length, lstm_hidden_size]
        lstm_out, _ = self.lstm(x)
        
        # output layer
        # output shape: [train_batch_size, output_size*forecast_horizon]
        output = self.fc(lstm_out[:, -1, :])
        
        # reshape output: [train_batch_size, forecast_horizon, output_size]
        output = torch.reshape(output, (-1,self.forecast_horizon, self.output_size))
        
        return output
    
    
def NLL_MDN_loss(output, target, num_gaussians):
    """NLL loss definition for MDN
    """
    
    # output shape: [train_batch_size, n_horizons, num_gaussians * 6] (mu_x, mu_y, sigma_x, sigma_y, rho, alpha for each Gaussian)
    # target shape: [train_batch_size, n_horizons, 2] (x, y)
    
    train_batch_size = target.size(0)
    forecast_horizon = target.size(1)
    
    # split the output into the parameters for each Gaussian
    mu_x = output[:,:, :num_gaussians]
    mu_y = output[:,:, num_gaussians:2*num_gaussians]
    sigma_x = torch.exp(output[:,:, 2*num_gaussians:3*num_gaussians])
    sigma_y = torch.exp(output[:,:, 3*num_gaussians:4*num_gaussians])
    rho = torch.tanh(output[:,:, 4*num_gaussians:5*num_gaussians])
    alpha = torch.softmax(output[:,:, 5*num_gaussians:], dim=-1)
    
    mixture = dist.Categorical(alpha)
    covs = torch.zeros(train_batch_size, forecast_horizon, num_gaussians, 2, 2).to(output.device)
    covs[:, :, :, 0, 0] = sigma_x ** 2
    covs[:, :, :, 0, 1] = rho * sigma_x * sigma_y
    covs[:, :, :, 1, 0] = rho * sigma_x * sigma_y
    covs[:, :, :, 1, 1] = sigma_y ** 2
    
    try:
        
        gaussians = dist.MultivariateNormal(torch.stack([mu_x, mu_y], dim=-1), covs)
        
    except:
        
        return None, True
        
    
    # compute the negative log likelihood loss for the mixture weights and the rest of the parameters
    mixture = dist.MixtureSameFamily(mixture, gaussians)
    params_loss = -mixture.log_prob(target).mean()
    loss = params_loss
    
    return loss, False