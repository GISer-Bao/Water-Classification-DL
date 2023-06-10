import math
import torch
from torch.nn import functional as F
import train_utils.distributed_utils as utils
from .loss_water import OhemCELoss, ECELoss

# 方案二，深监督
def criterion_train(inputs, target, n_min: int=58000, num_classes: int = 2, ignore_index: int = -100):
    
    criteria = ECELoss(thresh=0.7, n_min=n_min, n_classes=num_classes, alpha=0.01, 
                        radius=1, beta=0.5, ignore_lb=ignore_index, mode='ohem').cuda()
    losses = [criteria(inputs[i], target) for i in range(len(inputs))]
    total_loss = sum(losses)
    return total_loss


# 方案一
def criterion_eval(inputs,target, n_min:int=58000, num_classes:int = 2, ignore_index:int = -100):
    
    criteria = ECELoss(thresh=0.7, n_min=n_min, n_classes=num_classes, alpha=0.01, 
                        radius=1, beta=0.5, ignore_lb=ignore_index, mode='ohem').cuda()
    loss = criteria(inputs, target)
    return loss


def evaluate(model, data_loader, device, n_min, num_classes=2):
    model.eval()
    mae_metric = utils.MeanAbsoluteError()
    f1_metric = utils.F1Score()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images, targets = images.to(device), targets.to(device)
            output = model(images)

            loss = criterion_eval(inputs=output, target=targets.squeeze(1).to(torch.long), 
                             n_min=n_min, num_classes=num_classes, ignore_index=255)
            metric_logger.update(loss=loss.item())
            
            # 新增，11.27
            output1 = torch.sum(output,dim=1).unsqueeze(dim=1)
            mae_metric.update(output1, targets)
            f1_metric.update(output1, targets)
            
        mae_metric.gather_from_all_processes()
        f1_metric.reduce_from_all_processes()

    # return mae_metric, f1_metric
    return metric_logger.meters["loss"].global_avg, mae_metric, f1_metric

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, n_min, num_classes=2, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            # loss = criterion(output, target)
            loss = criterion_eval(inputs=output[0], target=target.squeeze(1).to(torch.long), 
                              n_min=n_min, num_classes=num_classes, ignore_index=255)
            
            # loss = criterion_train(inputs=output, target=target.squeeze(1).to(torch.long), 
            #                   n_min=n_min, num_classes=num_classes, ignore_index=255)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-4):
    params_group = [{"params": [], "weight_decay": 0.},  # no decay
                    {"params": [], "weight_decay": weight_decay}]  # with decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            # bn:(weight,bias)  conv2d:(bias)  linear:(bias)
            params_group[0]["params"].append(param)  # no decay
        else:
            params_group[1]["params"].append(param)  # with decay

    return params_group
