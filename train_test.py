# import wandb  # disabled: logs to file only
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import utils
import qm9.utils as qm9utils
from qm9 import losses
import logging
import time
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist, scaler=None, total_epochs=None):
    model_dp.train()
    model.train()
    # On first epoch where EMA is used, sync EMA from warmed-up model (don't start from cold init)
    ema_start = getattr(args, 'ema_start_epoch', 0)
    if args.ema_decay > 0 and epoch == ema_start:
        model_ema.load_state_dict(model.state_dict())
        log.info('EMA synced from model at start of epoch %d (warmup done)', epoch + 1)
    nll_epoch = []
    clip_count = 0
    max_clipped_norm = 0.0
    n_iterations = len(loader)
    epoch_desc = f"Epoch {epoch + 1}/{total_epochs}" if total_epochs is not None else f"Epoch {epoch + 1}"
    pbar = tqdm(loader, desc=epoch_desc, unit="batch", leave=True)
    for i, data in enumerate(pbar):
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        x = remove_mean_with_mask(x, node_mask)

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        optim.zero_grad()

        use_amp = getattr(args, 'use_amp', False) and device.type == 'cuda'
        amp_dtype = torch.bfloat16 if getattr(args, 'amp_dtype', 'fp16') == 'bfloat16' else torch.float16
        if use_amp:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                        x, h, node_mask, edge_mask, context)
                loss = nll + args.ode_regularization * reg_term
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad:
                    scaler.unscale_(optim)
                    grad_norm, was_clipped = utils.gradient_clipping(model, gradnorm_queue)
                    if was_clipped:
                        clip_count += 1
                        max_clipped_norm = max(max_clipped_norm, grad_norm)
                else:
                    grad_norm = 0.0
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad:
                    grad_norm, was_clipped = utils.gradient_clipping(model, gradnorm_queue)
                    if was_clipped:
                        clip_count += 1
                        max_clipped_norm = max(max_clipped_norm, grad_norm)
                else:
                    grad_norm = 0.0
                optim.step()
        else:
            nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                    x, h, node_mask, edge_mask, context)
            loss = nll + args.ode_regularization * reg_term
            loss.backward()
            if args.clip_grad:
                grad_norm, was_clipped = utils.gradient_clipping(model, gradnorm_queue)
                if was_clipped:
                    clip_count += 1
                    max_clipped_norm = max(max_clipped_norm, grad_norm)
            else:
                grad_norm = 0.0
            optim.step()

        # Update EMA if enabled and we are at or past the start epoch.
        ema_start = getattr(args, 'ema_start_epoch', 0)
        if args.ema_decay > 0 and epoch >= ema_start:
            ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            pbar.set_postfix(loss=f"{loss.item():.2f}", nll=f"{nll.item():.2f}", reg=f"{reg_term.item():.1f}", gn=f"{grad_norm:.1f}")
        log_every = getattr(args, 'log_metrics_every', 100)
        if i % log_every == 0:
            log.info('metrics batch_nll=%.4f loss=%.4f reg_term=%.2f grad_norm=%.2f iter=%d epoch=%d',
                     nll.item(), loss.item(), reg_term.item(), grad_norm, i, epoch + 1)
        nll_epoch.append(nll.item())
        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0):
            start = time.time()
            if len(args.conditioning) > 0:
                save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch)
            save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
                                  batch_id=str(i))
            sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
                                            prop_dist, epoch=epoch)
            log.info('Sampling took %.2f seconds', time.time() - start)

            vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=None)
            vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=None)
            if len(args.conditioning) > 0:
                vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
                                    wandb=None, mode='conditional')
        if args.break_train_epoch:
            break
    log.info('')
    if clip_count > 0:
        log.info('gradient_clips epoch=%d count=%d max_norm=%.2f', epoch + 1, clip_count, max_clipped_norm)
    log.info('metrics train_epoch_nll=%.4f epoch=%d', np.mean(nll_epoch), epoch + 1)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test'):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * args.augment_noise

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            nll, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                    node_mask, edge_mask, context)
            nll_epoch += nll.item() * batch_size
            n_samples += batch_size

    mean_nll = nll_epoch / n_samples
    log.info('%s NLL epoch=%d final=%.4f (n_samples=%d)', partition, epoch, mean_nll, n_samples)
    return mean_nll


def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    one_hot, charges, x = sample_chain(args=args, device=device, flow=model,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain')

    return one_hot, charges, x


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        log.debug("Generated molecule positions: %s", x[:-1, :, :].shape)
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, dataset_info,
                          batch_size * counter, name='molecule')


def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=100, dataset_smiles_list=None):
    """Generate n_samples molecules in batches of batch_size, then analyze stability."""
    batch_size = min(batch_size, n_samples)
    n_batches = n_samples // batch_size
    assert n_samples % batch_size == 0, 'n_stability_samples must be divisible by batch_size (default 100)'
    log.info('Analyzing molecule stability at epoch %s... (%d samples in %d batches of %d)',
             epoch, n_samples, n_batches, batch_size)
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info,
                                                                  dataset_smiles_list=dataset_smiles_list)

    log.info('metrics stability: %s', validity_dict)
    if rdkit_tuple is not None:
        log.info('metrics validity=%.4f uniqueness=%.4f novelty=%.4f',
                 rdkit_tuple[0][0], rdkit_tuple[0][1], rdkit_tuple[0][2])
    return validity_dict


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x
