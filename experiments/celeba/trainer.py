import os
from argparse import ArgumentParser
import logging
import sys
import numpy as np
import time
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
            dataset=train_set, batch_size=bs, shuffle=True, num_workers=2)
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

    # Logging loss format
    logger.info(
        f"Epoch | Train Loss | Val F1 Mean | Test F1 Mean | Time per epoch (min)"
    )

    for epoch in range(epochs):
        # training
        model.train()
        t0 = time.time()
        for j, (x, y) in enumerate(train_loader):
            if j == 0:
                break
            x = x.to(device)
            y = [y_.to(device) for y_ in y]
            y_ = model(x)
            losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
            optimizer.zero_grad()
            if args.epoch:
                update = (j == len(train_loader) - 1)
            else:
                update = True
            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                lr=lr,
                update=update
            )
            loss_list.append(losses.detach().cpu())
            optimizer.step()
            if "famo" in args.method:
                with torch.no_grad():
                    y_ = model(x)
                    new_losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
                    weight_method.method.update(new_losses.detach())
        t1 = time.time()

        model.eval()
        # validation
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

        # testing
        metric.reset()
        with torch.no_grad():
            for k, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = [y_.to(device) for y_ in y]
                y_ = model(x)
                if k == 0:
                    losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
                else:
                    losses += torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
                metric.incr(y_, y)
        test_f1 = metric.result()
        metrics[epoch] = test_f1
        print(f'Untrained metrics: {test_f1}')
        print(f'Untrained test losses: {losses/k}')
        1/0

        t2 = time.time()

        # Logging metrics
        logger.info(
            f"{epoch+1:03d} | Train Loss: {losses} | Val F1: {val_f1} | Test F1: {test_f1} | Time: {(t2-t0)/60:.2f}".replace("\n", "")
        )

        # Save results
        name = f"{args.method}_{args.alpha}_{args.norm}_{args.epoch}_{args.seed}"
        torch.save({"metric": metrics, "losses": loss_list, "best_epoch": best_epoch}, f"./save/{name}.stats")


if __name__ == "__main__":
    parser = ArgumentParser("Celeba", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=3e-4,
        n_epochs=15,
        batch_size=256,
        seed=10000,
        method='fairmtl',
        alpha=1.0
    )
    parser.add_argument("--norm", type=str2bool, default=False, help="Normalize weights for FairMTL?")
    parser.add_argument("--epoch", type=str2bool, default=False, help="Update weights at each epoch (True) or each batch (False) for FairMTL?")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"trainlogs/{args.method}_{args.alpha}_{args.norm}_{args.epoch}_{args.seed}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    # set seed
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    main(path=args.data_path,
         lr=args.lr,
         bs=args.batch_size,
         device=device)
