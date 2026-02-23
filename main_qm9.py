# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import logging
import utils
utils.setup_logging()
log = logging.getLogger(__name__)
import argparse
# import wandb  # disabled: use --no_wandb or log file only
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9 import dataset
from qm9.models import get_optim, get_model
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from tqdm import tqdm
from train_test import train_epoch, test, analyze_and_save

parser = argparse.ArgumentParser(description='E3Diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')
parser.add_argument('--use_ddim', type=eval, default=False,
                    help='Use DDIM for sampling (inference/stability): deterministic, fewer steps')
parser.add_argument('--sampling_steps', type=int, default=None,
                    help='Number of denoising steps when sampling (default: diffusion_steps). With use_ddim, e.g. 50.')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=128,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory (legacy pipeline)')
parser.add_argument('--use_pyg_qm9', type=eval, default=True,
                    help='Use PyTorch Geometric QM9 (default True). Set False for legacy figshare pipeline.')
parser.add_argument('--qm9_root', type=str, default='./data/qm9',
                    help='Root dir for PyG QM9 download (used when use_pyg_qm9=True)')
parser.add_argument('--qm9_train_size', type=int, default=100_000,
                    help='Max training samples for PyG QM9')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1,
                    help='Update tqdm postfix every N batches')
parser.add_argument('--log_metrics_every', type=int, default=100,
                    help='Log batch metrics to file every N batches (default 100)')
# parser.add_argument('--wandb_usr', type=str)
# parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
# parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for the dataloader')
parser.add_argument('--prefetch_factor', type=int, default=2, help='Batches to prefetch per worker (used when num_workers > 0)')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--ema_start_epoch', type=int, default=1,
                    help='Start updating EMA from this epoch (0-based). Default 1 = after first epoch.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
# LoRA, checkpointing, mixed precision
parser.add_argument('--lora_rank', type=int, default=0,
                    help='LoRA rank; 0 = full model fine-tune, >0 = LoRA (requires --pretrained_model_path)')
parser.add_argument('--lora_alpha', type=float, default=None,
                    help='LoRA alpha (default: lora_rank)')
parser.add_argument('--pretrained_model_path', type=str, default=None,
                    help='Path to pretrained checkpoint (dir or .npy). Required when lora_rank > 0.')
parser.add_argument('--use_checkpointing', type=eval, default=False,
                    help='Use gradient checkpointing in EGNN blocks to save memory')
parser.add_argument('--use_amp', type=eval, default=False,
                    help='Use automatic mixed precision (AMP) for training')
parser.add_argument('--amp_dtype', type=str, default='fp16', choices=['fp16', 'bfloat16'],
                    help='AMP dtype: fp16 (default) or bfloat16. bfloat16 is often more stable (fewer inf/nan).')
parser.add_argument('--optimizer', type=str, default='adamw',
                    help='Optimizer: adamw | adam8bit (requires bitsandbytes)')
parser.add_argument('--config', type=str, default=None,
                    help='Path to YAML config file (overrides defaults)')
parser.add_argument('--preset', type=str, default=None,
                    help='Preset name (e.g. edm_qm9) -> configs/<preset>.yaml')
args = parser.parse_args()

# Load config from file or preset
if args.preset:
    import os
    config_path = os.path.join(os.path.dirname(__file__), 'configs', args.preset + '.yaml')
    if os.path.isfile(config_path):
        args.config = config_path
if args.config:
    try:
        import yaml
    except ImportError:
        raise ImportError('Install PyYAML (pip install pyyaml) to use --config/--preset')
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if cfg:
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
        log.info('Loaded config: %s', args.config)

if args.resume is None:
    from datetime import datetime
    args.exp_name = args.exp_name + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

dataset_info = get_dataset_info(args.dataset, args.remove_h)

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
# args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    # wandb_usr = getattr(args, 'wandb_usr', None)
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    # args.wandb_usr = wandb_usr

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    log.info('Args: %s', args)

utils.create_folders(args)

# All logs to file
log_dir = join('outputs', args.exp_name)
log_file = join(log_dir, 'train.log')
utils.add_log_file(log_file)
log.info('Logging to %s', log_file)
log.info('Config: %s', getattr(args, 'config', None))
log.info('Args: %s', args)

# Wandb disabled (uncomment to enable)
# if args.no_wandb:
#     mode = 'disabled'
# else:
#     mode = 'online' if args.online else 'offline'
# kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion', 'config': args,
#           'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
# wandb.init(**kwargs)
# wandb.save('*.txt')

# Retrieve QM9 dataloaders (once; not recreated each epoch)
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

data_dummy = next(iter(dataloaders['train']))

# Precompute train SMILES once for RDKit novelty metric (avoids reloading QM9 inside stability analysis)
train_smiles_for_metrics = None
if 'qm9' in args.dataset:
    try:
        from qm9.rdkit_functions import get_train_smiles_for_metrics
        train_smiles_for_metrics = get_train_smiles_for_metrics(dataloaders['train'], dataset_info)
    except Exception as e:
        log.warning('Could not precompute train SMILES for metrics: %s', e)


if len(args.conditioning) > 0:
    log.info('Conditioning on %s', args.conditioning)
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf


# Create EGNN flow
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
if prop_dist is not None:
    prop_dist.set_normalizer(property_norms)

lora_rank = getattr(args, 'lora_rank', 0)
pretrained_path = getattr(args, 'pretrained_model_path', None)
if lora_rank > 0:
    if not pretrained_path:
        raise ValueError('When lora_rank > 0, --pretrained_model_path is required (path to dir or generative_model.npy)')
    import os
    from egnn.lora import inject_lora, freeze_base_lora
    path = pretrained_path
    if os.path.isdir(path):
        fn = 'generative_model_ema.npy' if getattr(args, 'ema_decay', 0) > 0 else 'generative_model.npy'
        path = join(path, fn)
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Pretrained checkpoint not found: {path}')
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    log.info('Loaded pretrained from %s', path)
    n_lora = inject_lora(model, rank=lora_rank, alpha=getattr(args, 'lora_alpha', None) or lora_rank)
    freeze_base_lora(model)
    log.info('LoRA injected: rank=%s, layers=%s, base frozen', lora_rank, n_lora)

model = model.to(device)
optim = get_optim(args, model)
# print(model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def main():
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'flow.npy'))
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:
        log.info('Training using %d GPUs', torch.cuda.device_count())
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    use_amp = getattr(args, 'use_amp', False) and args.cuda
    amp_dtype = getattr(args, 'amp_dtype', 'fp16')
    scaler = torch.cuda.amp.GradScaler() if use_amp and amp_dtype == 'fp16' else None
    if use_amp:
        log.info('Training with mixed precision (AMP), dtype=%s', amp_dtype)
    best_nll_val = 1e8
    best_nll_test = 1e8
    epoch_range = range(args.start_epoch, args.n_epochs)
    for epoch in tqdm(epoch_range, desc="Epochs", unit="ep", total=args.n_epochs):
        start_epoch = time.time()
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist, scaler=scaler,
                    total_epochs=args.n_epochs)
        log.info("Epoch %d took %.1f s.", epoch + 1, time.time() - start_epoch)

        if epoch % args.test_epochs == 0:
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                model_info = model.log_info()
                log.info('Model info: %s', model_info)
                # wandb.log(model_info, commit=True)

            if not args.break_train_epoch:
                analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                                 dataset_info=dataset_info, device=device,
                                 prop_dist=prop_dist, n_samples=args.n_stability_samples,
                                 dataset_smiles_list=train_smiles_for_metrics)
            nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms)
            nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Test', device=device, dtype=dtype,
                            nodes_dist=nodes_dist, property_norms=property_norms)

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

                if args.save_model:
                    utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                    utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
                    with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                        pickle.dump(args, f)
            log.info('Val loss: %.4f \t Test loss: %.4f', nll_val, nll_test)
            log.info('Best val loss: %.4f \t Best test loss: %.4f', best_nll_val, best_nll_test)
            log.info('metrics epoch=%d nll_val=%.4f nll_test=%.4f best_nll_val=%.4f best_nll_test=%.4f',
                     epoch + 1, nll_val, nll_test, best_nll_val, best_nll_test)
            # wandb.log({"Val loss ": nll_val}, commit=True)
            # wandb.log({"Test loss ": nll_test}, commit=True)
            # wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


if __name__ == "__main__":
    main()
