import os
from argparse import ArgumentParser
import logging
import sys
import numpy as np
import time
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.celeba.data import CelebaDataset
from experiments.celeba.models import Network
from utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from methods.weight_methods import WeightMethods

# Alpha-Fairness Function
def alpha_fairness(x, alpha):
    if alpha == 1:
        return torch.log(x)  # Special case for alpha=1 (logarithmic fairness)
    return (x**(1 - alpha) - 1) / (1 - alpha)


class CelebaMetrics():
    """
    CelebA metric accumulator.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.tp = 0.0 
        self.fp = 0.0 
        self.fn = 0.0 
        
    def incr(self, y_preds, ys):
        # y_preds: [ y_pred (batch, 1) ] x 40
        # ys     : [ y_pred (batch, 1) ] x 40
        y_preds  = torch.stack(y_preds).detach() # (40, batch, 1)
        ys       = torch.stack(ys).detach()      # (40, batch, 1)
        y_preds  = y_preds.gt(0.5).float()
        self.tp += (y_preds * ys).sum([1,2]) # (40,)
        self.fp += (y_preds * (1 - ys)).sum([1,2])
        self.fn += ((1 - y_preds) * ys).sum([1,2])
                
    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall    = self.tp / (self.tp + self.fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        return f1.cpu().numpy()


def main(path, lr, bs, device):
    # we only train for specific task
    model = Network().to(device)
    
    train_set = CelebaDataset(data_dir=path, split='train')
    val_set   = CelebaDataset(data_dir=path, split='val')
    test_set  = CelebaDataset(data_dir=path, split='test')

    train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=bs, num_workers=2, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=bs, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs    = args.n_epochs

    metrics   = np.zeros([epochs, 40], dtype=np.float32) # test_f1
    metric    = CelebaMetrics()
    loss_fn   = torch.nn.BCELoss()

    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    weight_method = WeightMethods(
        args.method, n_tasks=40, device=device, **weight_methods_parameters[args.method]
    )

    best_val_f1 = 0.0
    best_epoch = None

    loss_list = []
    util_list = []
    sum_utils = torch.zeros(40).to(device)
    fairness_list = []
    times_list = []

    # Logging loss format
    logger.info(
        f"Epoch | Train Loss | Val F1 Mean | Test F1 Mean | Time per epoch (min)"
    )
    loss_max = args.loss_max
    iteration = 1

    for epoch in range(epochs):
        # training
        epoch_loss = torch.zeros(40).to(device)
        model.train()
        t0 = time.time()
        for j, (x, y) in enumerate(train_loader):
            # if j == 50:
            #     break
            priorities = torch.ones(40).to(device)
            if j%2 == 0:
                priorities[0] = 10
            # else:
            #     priorities[1] = 10
            # print(priorities)
            priorities = priorities/max(priorities)
            x = x.to(device)
            y = [y_.to(device) for y_ in y]
            y_ = model(x)
            losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
            epoch_loss += losses.detach()
            losses = losses * priorities
            optimizer.zero_grad()
            utils = priorities*loss_max - losses.detach() + 1
            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                lr=lr,
                utilities=utils
            )
            loss_list.append(losses.detach().cpu())
            util_list.append(utils)
            sum_utils += utils
            fairness = alpha_fairness(sum_utils/iteration, args.alpha).sum().cpu()
            if tb_writer is not None:
                    tb_writer.add_scalar(f"meta/fairness", fairness.item(), iteration)
                    for user in range(40):
                        tb_writer.add_scalar(f"meta/avg_util_{user}", sum_utils[user].item()/iteration, iteration)
            # print(j, fairness)
            fairness_list.append(fairness)
            optimizer.step()
            if "famo" in args.method:
                with torch.no_grad():
                    y_ = model(x)
                    new_losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
                    new_losses = new_losses * priorities
                    weight_method.method.update(new_losses.detach())
            iteration = iteration + 1
        loss_max = epoch_loss / j+1
        t1 = time.time()
        times_list.append(t1-t0)

        # model.eval()
        # # validation
        # metric.reset()
        # with torch.no_grad():
        #     for k, (x, y) in enumerate(val_loader):
        #         if k == 0:
        #             break
        #         x = x.to(device)
        #         y = [y_.to(device) for y_ in y]
        #         y_ = model(x)
        #         losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
        #         metric.incr(y_, y)
        # val_f1 = metric.result()
        # if val_f1.mean() > best_val_f1:
        #     best_val_f1 = val_f1.mean()
        #     best_epoch = epoch

        # # testing
        # metric.reset()
        # with torch.no_grad():
        #     for k, (x, y) in enumerate(test_loader):
        #         x = x.to(device)
        #         y = [y_.to(device) for y_ in y]
        #         y_ = model(x)
        #         if k == 0:
        #             losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
        #         else:
        #             losses += torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
        #         metric.incr(y_, y)
        # test_f1 = metric.result()
        # metrics[epoch] = test_f1

        # t2 = time.time()

        # # Logging metrics
        # logger.info(
        #     f"{epoch+1:03d} | Train Loss: {losses} | Val F1: {val_f1} | Test F1: {test_f1} | Time: {(t2-t0)/60:.2f}".replace("\n", "")
        # )

        # Save results
        name = f"{args.method}_{args.alpha}_{args.seed}_{args.main_task}"
        torch.save({"metric": metrics, "losses": loss_list, "best_epoch": best_epoch, "utilities": util_list, "fairness": fairness_list, "times": times_list}, f"./save/{name}.stats")

        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser("Celeba", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=3e-4,
        n_epochs=1,
        batch_size=10,
        seed=0,
        method='memtl',
        alpha=2.0,
        loss_max=np.log(2)
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"trainlogs/{args.method}_{args.alpha}_{args.seed}_{args.main_task}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    writer_path = f'runs/{args.method}_{args.alpha}_{args.seed}_{args.main_task}'
    print(writer_path)
    tb_writer = SummaryWriter(writer_path, purge_step=0)
    logger = logging.getLogger(__name__)
    # set seed
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    main(path=args.data_path,
         lr=args.lr,
         bs=args.batch_size,
         device=device)
