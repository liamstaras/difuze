import torch
import numpy as np
from support import NoiseSchedule, NumpyMetric
from metrics.support import Statistic
from data import NpyDataset, NpySaver, TifSaver
from metrics.cosmology import PixelCounts, PeakCounts, PowerSpectrum
from metrics.metrics import RelativeDifference
from diffusion import DiffusionFramework
from palette.models import PaletteModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from socket import gethostname
import os
import cmocean

# determine whether we will use the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# establish base path
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
base_path = os.path.join('runs', timestamp+'_'+gethostname())

# create the model
model = PaletteModel(
    image_size=128,
    in_channel=2,
    out_channel=1,
    inner_channel=64,
    channel_mults=(1,2,4,8),
    attn_res=(16,),
    num_head_channels=32,
    res_blocks=2,
    dropout=0.2
)

# create the loss function, optimizer and scheduler
training_loss_function = torch.nn.L1Loss()
optimizer = torch.optim.Adam(
    params=list(filter(lambda p: p.requires_grad, model.parameters())), 
    lr=1e-6, 
    weight_decay=0
)
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=optimizer,
    gamma=0.9
)

# make noise schedules
training_noise_schedule = NoiseSchedule(2000, 1e-6, 0.01, np.linspace)
inference_noise_schedule = NoiseSchedule(300, 1e-4, 0.09, np.linspace)

# make datasets
training_dataset = NpyDataset(
    'data/full-log-clean.npy',
    gt_index=1,
    cond_index=0,
    stop_index=0.9
)
evaulation_dataset = NpyDataset(
    'data/full-log-clean.npy',
    gt_index=1,
    cond_index=0,
    start_index=0.9
)

# prepare evaluation metrics

statistics = [
    PixelCounts(), PeakCounts(), PowerSpectrum()
]
evaluation_metrics = [
    NumpyMetric(
        RelativeDifference(
            statistic
        )
    ) for statistic in statistics]

# make dataloaders
training_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
evaluation_dataloader = DataLoader(evaulation_dataset, batch_size=32, shuffle=False)

# prepare tensorboard
summary_writer = SummaryWriter(log_dir=os.path.join(base_path, 'tensorboard'))

# make function for generating images
def tensor_to_image_cmocean(tensor):
    cmin = -2
    cmax = 2
    normalized = (torch.clamp(tensor, cmin, cmax)-cmin)/(cmax-cmin) # now in range (0,1)
    image = cmocean.cm.deep_r(normalized.cpu().numpy())
    # roll axes and cut alpha channel
    return image.transpose((2,0,1))[:3]

# specify functions to save outputs
save_functions = [
    NpySaver(),
    TifSaver(tensor_to_image_cmocean)
]
visual_function = tensor_to_image_cmocean

# wrap everything up in the diffusion framework
framework = DiffusionFramework(
    device=device,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    training_dataloader=training_dataloader,
    training_noise_schedule=training_noise_schedule,
    training_loss_function=training_loss_function,
    evaluation_dataloader=evaluation_dataloader,
    inference_noise_schedule=inference_noise_schedule,
    evaluation_metrics=evaluation_metrics,
    base_path=base_path,
    summary_writer=summary_writer,
    save_functions=save_functions,
    visual_function=visual_function
)

framework.main_training_loop()