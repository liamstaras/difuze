import torch
import numpy as np
from support import NoiseSchedule, NumpyMetric
from data import NpyDataset, NpySaver, TifSaver
from metrics.cosmology import PixelCounts, PeakCounts, PowerSpectrum
from metrics.metrics import RelativeDifference
from diffusion import DiffusionFramework
from palette.models import PaletteModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cmocean

# determine whether we will use the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    lr=5e-6, 
    weight_decay=0
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    factor=0.1
)

# make noise schedules
training_noise_schedule = NoiseSchedule(2000, 1e-6, 0.01, np.linspace)
inference_noise_schedule = NoiseSchedule(1000, 1e-4, 0.09, np.linspace)

# make datasets
training_dataset = NpyDataset(
    '/home/liam/Documents/Code/Genève/Davide-data/full-log-clean.npy',
    gt_index=1,
    cond_index=0,
    stop_index=0.9
)
evaulation_dataset = NpyDataset(
    '/home/liam/Documents/Code/Genève/Davide-data/full-log-clean.npy',
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
training_dataloader = DataLoader(training_dataset, batch_size=4, shuffle=True)
evaluation_dataloader = DataLoader(evaulation_dataset, batch_size=4, shuffle=False)

# prepare tensorboard
summary_writer = SummaryWriter()

# make function for generating images
def tensor_to_image_cmocean(tensor):
    normalized = torch.clamp(tensor, -4, 4)/4 # now in range (-1,1)
    image = cmocean.cm.deep_r(normalized.cpu().numpy())
    return image

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
    summary_writer=summary_writer,
    save_functions=save_functions,
    visual_function=visual_function
)

framework.main_training_loop()