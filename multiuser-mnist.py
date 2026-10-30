import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToPILImage
from itertools import product
import os
import higher
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from utils import (
    common_parser,
    str2bool,
)
from methods.weight_methods import WeightMethods
from typing import Iterator
from itertools import chain


def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU
        torch.cuda.manual_seed_all(seed)  # All GPUs
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior for CNNs
        torch.backends.cudnn.benchmark = False  # Disable auto-tuner for deterministic results

# Alpha-Fairness Function
def alpha_fairness(x, alpha):
    if alpha == 1:
        return torch.log(x)  # Special case for alpha=1 (logarithmic fairness)
    return (x**(1 - alpha) - 1) / (1 - alpha)

class MultiUserModel(nn.Module):
    def __init__(self):
        super(MultiUserModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, return_indices=True)[0]
        x = torch.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, return_indices=True)[0]
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Meta part of the dataset (with 56 different transformation):
# Transform to overlay MNIST digit on a rainbow background
class RainbowTransform:
    def __init__(self, color=None):
        """
        Initialize the transform with a specific color as a string or allow random selection.
        :param color: String name of the color (e.g., "red", "blue"). 
                      If None, a random color from the rainbow will be selected.
        """
        self.color_map = {
            "red": (255, 0, 0),
            "orange": (255, 165, 0),
            "yellow": (255, 255, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "violet": (139, 0, 255),
            "black": (0, 0, 0)
        }
        self.color = color

    def __call__(self, image):
        """
        Apply the rainbow background transform.
        :param image: Grayscale MNIST digit (PIL Image).
        :return: MNIST digit overlaid on a colorful background (PIL Image).
        """
        if self.color:
            # Get the corresponding RGB value from the color map
            if self.color not in self.color_map:
                raise ValueError(f"Color '{self.color}' is not valid. Choose from {list(self.color_map.keys())}.")
            background_color = self.color_map[self.color]
        else:
            # Select a random color from the color map
            background_color = self.color_map[np.random.choice(list(self.color_map.keys()))]
        
        # Create a solid background with the selected color
        background = Image.new("RGB", image.size, background_color)
        
        # Convert the MNIST digit to RGBA to enable alpha blending
        digit = image.convert("RGBA")
        
        # Add transparency to the digit for overlay
        digit_data = np.array(digit)
        # digit_data[digit_data > 50] = 255
        # digit_data[digit_data < 50] = 1
        digit_data[:, :, 3] = (digit_data[:, :, 0])  # Set transparency based on pixel intensity
        digit = Image.fromarray(digit_data, mode="RGBA")
        
        # Overlay the digit onto the background
        combined = Image.alpha_composite(background.convert("RGBA"), digit)
        return combined.convert("RGB")

# Example transform for scaling or orientation tasks
class ScaleTransform:
    def __init__(self, scale_factor, background_color):
        """
        Initialize the scale transform.
        :param scale_factor: Scaling factor for the image.
        :param background_color: Color to use for padding (default: white).
        """
        self.scale_factor = scale_factor
        self.color_map = {
            "red": (255, 0, 0),
            "orange": (255, 165, 0),
            "yellow": (255, 255, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "violet": (139, 0, 255),
            "black": (0, 0, 0)
        }
        if background_color not in self.color_map:
            raise ValueError(f"Background color '{background_color}' is not valid. Choose from {list(self.color_map.keys())}.")
        self.background_color = self.color_map[background_color]


    def __call__(self, image):
        """
        Apply the scaling transform while keeping the original size by padding.
        :param image: Input PIL image.
        :return: Scaled and padded PIL image.
        """
        # Calculate new size after scaling
        new_size = (int(image.size[0] * self.scale_factor), int(image.size[1] * self.scale_factor))
        
        # Resize the image to the new size
        scaled_image = image.resize(new_size, Image.Resampling.NEAREST)
        
        # Create a blank image with the original size and the background color
        padded_image = Image.new("RGB", image.size, self.background_color)
        
        # Calculate top-left coordinates to center the scaled image
        top_left_x = (image.size[0] - new_size[0]) // 2
        top_left_y = (image.size[1] - new_size[1]) // 2
        
        # Paste the scaled image onto the padded image
        padded_image.paste(scaled_image, (top_left_x, top_left_y))
        
        return padded_image

class RotateTransform:
    def __init__(self, angle, background_color):
        self.angle = angle
        self.color_map = {
            "red": (255, 0, 0),
            "orange": (255, 165, 0),
            "yellow": (255, 255, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "violet": (139, 0, 255),
            "black": (0, 0, 0)
        }
        if background_color not in self.color_map:
            raise ValueError(f"Background color '{background_color}' is not valid. Choose from {list(self.color_map.keys())}.")
        self.background_color = self.color_map[background_color]

    def __call__(self, image):
        # Rotate the image by the given angle
        return image.rotate(self.angle, fillcolor=self.background_color)
    
def meta_transform(seed=0): 
    to_pil = ToPILImage()
    # Meta-task parameters
    colors = ["red", "orange", "yellow", "green", "blue", "violet", "black"]  # 7 colors
    scales = [0.5, 1]  # 2 scales
    rotations = [0, 90, 180, 270]  # 4 rotations

    # Generate all meta-task combinations (7 × 2 × 4 = 56 tasks)
    task_combinations = list(product(colors, scales, rotations))
    num_meta_tasks = len(task_combinations)

    # List to hold datasets for each task
    task_datasets = []
    val_datasets = []
    save_path = f'./cached_mnist_{seed}'

    if not os.path.exists(save_path):
        transform = transforms.Compose([transforms.ToTensor()])
        mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        # Divide the MNIST dataset into 56 parts
        chunk_size = len(mnist) // num_meta_tasks
        indices = list(range(len(mnist)))
        random.shuffle(indices)
        # os.mkdir(save_path)
        # with open(f"{save_path}/.gitignore", "a") as f:
        #     f.write("*")
        for i, (color, scale, rotation) in enumerate(task_combinations):
            # Get indices for the current chunk
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size # if i < num_meta_tasks - 1 else len(mnist)
            chunk_indices = indices[start_idx:end_idx]

            # Define the transformation for this task
            task_transform = transforms.Compose([
                RainbowTransform(color=color),       # Background color
                ScaleTransform(scale_factor=scale, background_color=color),  # Scaling
                RotateTransform(angle=rotation, background_color=color),     # Rotation
                transforms.ToTensor(),               # Convert to tensor
                transforms.Normalize((0.5,), (0.5,)) # Normalize
            ])
            task_dataset = Subset(mnist, chunk_indices)
            # Apply the task-specific transform
            transformed_task_dataset = [(task_transform(to_pil(image[0])), image[1]) for image in task_dataset]          
            # Get subset from applied transform
            task_datasets.append(transformed_task_dataset)
            # torch.save(transformed_task_dataset, f"{save_path}/train_{i}.pt")
    else:
        for i in range(num_meta_tasks):
            task_datasets.append(torch.load(f'{save_path}/train_{i}.pt'))
    return task_datasets

def show_samples(datasets, img_id):
    # Setup for RGB normalization reversal
    def unnormalize(img_tensor):
        img = img_tensor.permute(1, 2, 0).numpy()  # CxHxW to HxWxC
        img = (img * 0.5) + 0.5  # Undo normalization
        img = np.clip(img, 0, 1)
        return img
    # Grid size
    n_rows = 7  # colors
    n_cols = 8  # 2 scales × 4 rotations
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Plot the same sample from each meta-task
    for i, dataset in enumerate(datasets):
        image, label = dataset[img_id]
        image = unnormalize(image)

        ax = axes[i]
        ax.imshow(image)
        ax.axis('off')
        # Optional: annotate with color/scale/rotation info
        # color_idx = i // (2 * 4)
        # scale_idx = (i // 4) % 2
        # rotation_idx = i % 4
        # ax.set_title(f"{colors[color_idx]}\nS:{scales[scale_idx]}, R:{rotations[rotation_idx]}",
        #              fontsize=8)
    # fig.suptitle(f"Label: {label}")
    plt.tight_layout()
    plt.show()

# Meta-update function
def meta_update(w, task_batch, update_lr, meta_optimizer, num_tasks, meta_dict, meta_steps=1, method='memtl', same=True, **kwargs):
    if method == 'cws':
        weights = 1/num_users * torch.ones(num_users).to(device)
    elif method == 'memtl':
        weights = kwargs['weights']
    elif method == 'short':
        weights = kwargs['weights']
    elif method == 'stl':
        weights = torch.zeros(num_users).to(device)
        weights[kwargs['stl_user']] = 1
    w_min = kwargs['w_min']*torch.ones(num_tasks).to(device)
    w_max = kwargs['w_max']*torch.ones(num_tasks).to(device)
    dual_lr = kwargs['dual_lr']
    dual_t = kwargs['dual_t']
    environment = kwargs['environment']
    sum_utils = kwargs['sum_utils']
    X = []
    y = []
    for user in range(num_users):
        X.append(task_batch[user][0].unsqueeze(0))
        y.append(task_batch[user][1])
    X = torch.cat(X).to(device)
    y = torch.Tensor(y).to(device)
    for i in range(meta_steps):
        with higher.innerloop_ctx(w, optim.SGD(w.parameters(), lr=update_lr), track_higher_grads=True) as (fmodel, diffopt):
            fmodel.train()
            # Use the same batch for the inner and outer update.
            # Task-specific updates using D_k^tr
            loss_train = 0
            # Inner-loop updates
            loss = criterion(fmodel(X), y).mean()
            diffopt.step(loss)
            loss_train += loss
            avg_loss_train = loss_train
            fmodel.eval()
            meta_loss_vec = criterion(fmodel(X), y)
            meta_loss = weights@(meta_loss_vec*environment)
            meta_loss_vec = meta_loss_vec.detach()
            # Back-prop through inner loop *once*
            meta_optimizer.zero_grad()  # Clear gradients for meta-update
            # with higher, we need to use following to obtain gradients:
            grads = torch.autograd.grad(meta_loss, fmodel.parameters(time=0), retain_graph=True)
            for param, grad in zip(w.parameters(), grads):
                param.grad = grad.detach()
            # before = [p.clone() for p in w.parameters()]
            meta_optimizer.step()               # update w
            # after = list(w.parameters())
            # print("Δ param norm:", torch.norm(torch.stack([(a - b).norm() for b, a in zip(before, after)])))
            # print(torch.nn.utils.clip_grad_norm_(w.parameters(), max_norm=1e9))
            # sanity = torch.stack([torch.dot((a-b).flatten(), grad.flatten()) for b, a, grad in zip(before, after, grads)])
            # print(torch.sum(sanity<=0)==sanity.numel())
            # for ours, we will only modify the weighting in this meta_loss calculation.
            util = (loss_max-meta_loss_vec)*environment + 1
            sum_utils += util
            dual_t += 1
            if method == 'memtl':
                weights = torch.clamp(sum_utils/dual_t, u_min, u_max)**(-alpha) # FTL
            elif method == 'short':
                weights = torch.clamp(util, u_min, u_max)**(-alpha)
            fairness = alpha_fairness(sum_utils/dual_t, alpha).sum()
            meta_dict['fairness'].append(fairness)
            if tb_writer is not None:
                tb_writer.add_scalar(f"meta/fairness", fairness, dual_t)
            for user in range(num_users):
                meta_dict[f'loss_{user}'].append(meta_loss_vec[user].item())
                meta_dict[f'util_{user}'].append(util[user].item())
                meta_dict[f'weight_{user}'].append(weights[user].item())
                if tb_writer is not None:
                    tb_writer.add_scalar(f"meta/loss_{user}", meta_loss_vec[user].item(), dual_t)
                    tb_writer.add_scalar(f"meta/util_{user}", util[user].item(), dual_t)
                    tb_writer.add_scalar(f"meta/avg_util_{user}", sum_utils[user].item()/dual_t, dual_t)
                    tb_writer.add_scalar(f"meta/weight_{user}", weights[user].item(), dual_t)
            # else:
            #     print(f"Meta Step {i+1}, Train Loss: {avg_loss_train:.4f}, Val Loss (L): {meta_loss_vec[0]:.4f}, Val Loss (R): {meta_loss_vec[1]:.4f}")
    return w, weights, dual_t, sum_utils

def train(model, meta_method, user_loaders, meta_optimizer, update_lr, meta_steps, grad_steps, meta_batch, weights, stl_user):
    sum_utils = torch.zeros(num_users).to(device)
    # Main loop
    t = 0
    dual_t = 0
    model.train()
    meta_dict = {'fairness': []}
    for user in range(num_users):
        meta_dict[f'loss_{user}'] = []
        meta_dict[f'util_{user}'] = []
        meta_dict[f'weight_{user}'] = []
    
    for ep in range(num_epochs):
        task_batches = []
        for task in range(len(user_loaders[0])):
            batches = []
            for user in range(num_users):
                batches.append(user_loaders[user][task])
            for batch in zip(*batches):
                t = t + 1
                if t%2 == 0:
                    environment = torch.Tensor([128, 64, 32, 16, 8, 4, 2, 1]).to(device)
                else:
                    environment = torch.Tensor([1, 64, 32, 16, 8, 4, 2, 128]).to(device)
                environment = environment/sum(environment)
                # batch is now a list with length num_users, each element is X, y tensors.
                model, weights, dual_t, sum_utils = meta_update(model, batch, update_lr, meta_optimizer, num_users, meta_dict, meta_steps=meta_steps, method=meta_method, w_min=w_min, w_max=w_max, dual_lr=dual_lr, weights=weights, dual_t=dual_t, environment=environment, sum_utils=sum_utils, stl_user=stl_user)
    torch.save({"meta": meta_dict}, results_path)

# Initialize and train the model
if __name__ == "__main__":
    parser = ArgumentParser("MNIST")
    parser.add_argument(
        "--method",
        type=str,
        default="memtl",
        choices=["cws", "memtl", "short", "stl"],
        help="Meta weighting method.",
    )
    parser.add_argument(
        "--user", type=int, default=0, help="Which user to train for when using STL."
    )
    parser.add_argument("--alpha", type=float, default=2.0, help="Fairness value.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--smooth", type=float, default=0.0, help="Label smoothing?")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--meta_lr", type=float, default=0.01, help="Learning rate for the meta-update.")
    parser.add_argument("--update_lr", type=float, default=0.01, help="Learning rate for the update-procedure.")
    parser.add_argument("--meta_steps", type=int, default=1, help="Number of meta steps.")
    parser.add_argument("--grad_steps", type=int, default=1, help="Number of grad steps.")
    parser.add_argument("--meta_batch", type=int, default=1, help="Data for meta update.")
    parser.add_argument("--tb", type=str2bool, default=False, help="Save tensorboard?")
    args = parser.parse_args()
    #
    meta_method = args.method
    user = args.user
    # Hyperparams for our method:
    alpha = args.alpha
    num_epochs = args.num_epochs
    num_users = 8
    u_min = 1
    loss_max = torch.log(torch.tensor(10)).item() # random guess+u_min
    u_max = (loss_max + u_min) # perfect prediction
    # General hyperparams for NNs:
    seed = args.seed
    smooth = args.smooth
    batch_size = args.batch_size
    meta_lr = args.meta_lr
    update_lr = args.update_lr
    meta_steps = args.meta_steps
    grad_steps = args.grad_steps
    meta_batch = args.meta_batch
    tb = args.tb
    # Calculations for our method:
    w_min = u_max**-alpha
    w_max = u_min**-alpha
    weights_init = w_min + (w_max - w_min)*np.random.rand()
    weights_init *= torch.ones(num_users)
    dual_lr = alpha/(u_min**(alpha+1))
    # 
    criterion = nn.CrossEntropyLoss(label_smoothing=smooth, reduction='none')
    set_seed(seed)
    model_name = f'{meta_method}_ml{meta_lr}_ul{update_lr}_sd{seed}_u{num_users}_a{alpha}_new'
    if meta_method == 'stl':
        model_name += f"_{int(user)}"
    writer_path = f'runs/{model_name}'
    print(writer_path)
    tb_writer = SummaryWriter(writer_path, purge_step=0) if tb else None
    results_path = f'save/{model_name}.stats'
    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    task_datasets = meta_transform(seed)
    random.shuffle(task_datasets)
    k = len(task_datasets) // num_users
    # Combine the lists into pairs and shuffle
    user_loaders = [task_datasets[i:i+k] for i in range(0, len(task_datasets), k)]
    # show_samples(task_datasets, 0)
    model = MultiUserModel().to(device)
    meta_optimizer = optim.SGD(model.parameters(), lr=meta_lr)
    train(model, meta_method, user_loaders, meta_optimizer, update_lr, meta_steps, grad_steps, meta_batch, weights_init.to(device), stl_user=user)
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()