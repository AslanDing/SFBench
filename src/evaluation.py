import torch
import numpy as np
from torch.nn.functional import mse_loss

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def SEDI(predicted_values, true_values, percentile):
    percentile = percentile.numpy()
    # num_percentile = percentile.shape[-1]

    gt_events_list = []
    pred_events_list = []

    # for i in range(num_percentile // 2):
    gt_events_low = (true_values < percentile[:, 0].reshape(1,-1,1))
    pred_events_low = np.sum(np.logical_and(predicted_values < percentile[:, 0].reshape(1,-1,1), gt_events_low), axis=(0, -1))

    gt_events_high = (true_values > percentile[:, 1].reshape(1,-1,1))
    pred_events_high = np.sum(
        np.logical_and(predicted_values > percentile[:, 1].reshape(1,-1,1), gt_events_high), axis=(0, -1))

    gt_events = np.sum(gt_events_low, axis=(0, -1)) + np.sum(gt_events_high, axis=(0, -1))

    gt_events_list.append(gt_events)
    pred_events_list.append(pred_events_high + pred_events_low)

    res = (np.array(pred_events_list)/(np.array(gt_events_list)+1E-4)).mean()
    return res

def interestingness_score(batch, mean, std):
    # mean = dataset.mean[:, None, 0].repeat(batch.num_graphs, 1).to(device)
    # std = dataset.std[:, None, 0].repeat(batch.num_graphs, 1).to(device)
    unnormalized_discharge = mean.view(1,-1,1) + std.view(1,-1,1) * batch
    B,N,T = unnormalized_discharge.shape
    unnormalized_discharge_min = unnormalized_discharge.transpose(0,1).contiguous().view(N,-1).min()
    unnormalized_discharge = unnormalized_discharge - unnormalized_discharge_min
    assert unnormalized_discharge.min() >= 0.0
    comparable_discharge = unnormalized_discharge / (mean - unnormalized_discharge_min)

    mean_central_diff = torch.gradient(comparable_discharge, dim=-1)[0].mean()
    trapezoid_integral = torch.trapezoid(comparable_discharge, dim=-1)

    score = 1e3 * (mean_central_diff ** 2) * trapezoid_integral
    assert not trapezoid_integral.isinf().any()
    assert not trapezoid_integral.isnan().any()
    return score.unsqueeze(-1)

# all tensor required
def NSE(pred,true,mean,std):
    # B,N,T = pred.shape

    model_mse = (pred-true)**2
    mean_mse = (true)**2

    weighted_nse = 1 - model_mse.sum(axis=-1) / (mean_mse+1E-8).sum(axis=-1)
    weighted_nse = weighted_nse.mean()
    return weighted_nse.cpu()

def cal_metrics(pred, true, mean, std, percents=None):
    metric_dict = {}

    
    if isinstance(pred,np.ndarray):
        pred_np = pred
        true_np = true
    else:
        pred_np = pred.cpu().detach().numpy()
        true_np = true.cpu().detach().numpy()

    if isinstance(mean,np.ndarray):
        mean_np = mean
        std_np = std
    else:
        mean_np = mean.cpu().detach().numpy()
        std_np = std.cpu().detach().numpy()

    mae = MAE(pred_np, true_np)
    metric_dict['mae'] = mae
    mse = MSE(pred_np, true_np)
    metric_dict['mse'] = mse
    rmse = RMSE(pred_np, true_np)
    metric_dict['rmse'] = rmse
    mape = MAPE(pred_np, true_np)
    metric_dict['mape'] = mape
    mspe = MSPE(pred_np, true_np)
    metric_dict['mspe'] = mspe

    if not percents is None:
        sedi_list = []

        if isinstance(percents[0],np.ndarray):
            pass
        else:
            percents = [percents[0].cpu().numpy(),percents[1].cpu().numpy(),percents[1].cpu().numpy()]
        for mask in percents:
            sedi = SEDI(pred_np, true_np, mask)
            sedi_list.append(sedi)
        metric_dict['sedi'] = sedi_list

    nse = NSE(pred_np,true_np,mean_np,std_np)
    metric_dict['nse'] = nse
    return metric_dict

