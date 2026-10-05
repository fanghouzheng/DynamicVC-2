import accelerate
import torch
import torch.nn as nn
import tyro
from datetime import timedelta
from config.config_flow import FlowConfig as Config
import torch.nn.functional as F
import time
from torch.utils.data import Dataset, DataLoader
import jax
import random
from src.data_process.data import Data, PerturbationDataset
from src.flow_matching.ot import OTPlanSampler
from src.flow_matching.path import AffineProbPath
from src.flow_matching.solver import ODESolver
from src.models.instantiate_model import instantiate_model
from src.tokenizer.gene_tokenizer import GeneVocab
from src.models.perturbation.moduls import PerturbationEmbedding
import pdb
import tqdm
from src.flow_matching.path.scheduler import CondOTScheduler
import scanpy as sc
import os
from src.data_process.utils import build_generated_anndata

import json
from accelerate import Accelerator,DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
import torchdiffeq
from tqdm import trange
import numpy as np
from cell_eval import MetricsEvaluator
import anndata as ad
import pandas as pd
from src.utils.utils import save_checkpoint, load_checkpoint, make_lognorm_poisson_noise, pick_eval_score, process_vocab, set_requires_grad_for_p_only, get_perturbation_emb

ot_sampler = OTPlanSampler(method="exact") 
path = AffineProbPath(scheduler=CondOTScheduler())

def gaussian_kernel(x, y, sigma=1.0):
    beta = 1.0 / (2.0 * sigma**2)
    dist = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-beta * dist)

def mmd_loss(pred, tgt, sigma=1.0):
    xx = gaussian_kernel(pred, pred, sigma).mean(dim=(1))
    yy = gaussian_kernel(tgt, tgt, sigma).mean(dim=(1))
    xy = gaussian_kernel(pred, tgt, sigma).mean(dim=(1))
    return (xx + yy - 2 * xy).mean()

def pairwise_sq_dists(X, Y):
    # X:[m,d], Y:[n,d] -> [m,n]
    return torch.cdist(X, Y, p=2)**2

@torch.no_grad()
def median_sigmas(X, scales=(0.5, 1.0, 2.0, 4.0)):
    Z = X
    D2 = pairwise_sq_dists(Z, Z)
    tri = D2[~torch.eye(D2.size(0), dtype=bool, device=D2.device)]
    m = torch.median(tri).clamp_min(1e-12)          
    s2 = torch.tensor(scales, device=Z.device) * m 
    sigmas = torch.sqrt(s2)                
    return [float(s.item()) for s in sigmas]

def mmd2_unbiased_multi_sigma(X, Y, sigmas):
    """
    """
    m, n = X.size(0), Y.size(0)
    Dxx = pairwise_sq_dists(X, X)   # [m,m]
    Dyy = pairwise_sq_dists(Y, Y)   # [n,n]
    Dxy = pairwise_sq_dists(X, Y)   # [m,n]

    vals = []
    for sigma in sigmas:
        beta = 1.0 / (2.0 * (sigma ** 2) + 1e-12)
        Kxx = torch.exp(-beta * Dxx)
        Kyy = torch.exp(-beta * Dyy)
        Kxy = torch.exp(-beta * Dxy)

        term_xx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1) + 1e-12)
        term_yy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1) + 1e-12)
        term_xy = Kxy.mean()  # / (m*n)
        vals.append(term_xx + term_yy - 2.0 * term_xy)

    return torch.stack(vals).mean()

def parse_trace_times(trace_times):
    times = []
    for item in str(trace_times).split(','):
        item = item.strip()
        if not item:
            continue
        t = float(item)
        if t <= 0.0 or t > 1.0:
            raise ValueError(f"trace_times must be in (0, 1], got {t}")
        times.append(t)
    if not times:
        raise ValueError("trace_times is empty")
    return sorted(set(times))

def sanitize_filename(name):
    return str(name).replace('/', '_').replace('\\', '_').replace(' ', '_')

def wait_for_marker(marker_path, poll_seconds=10):
    while not os.path.exists(marker_path):
        time.sleep(poll_seconds)

def select_trace_gene_panel(source, target, gene_ids_trace, gene_names):
    n_genes = source.shape[1]
    if config.trace_gene_panel == "all":
        gene_idx = torch.arange(n_genes, device=source.device)
    elif config.trace_gene_panel == "infer_top_gene":
        gene_idx = torch.arange(min(config.infer_top_gene, n_genes), device=source.device)
    else:
        raise ValueError(f"Unsupported trace_gene_panel: {config.trace_gene_panel}")

    source = source[:, gene_idx]
    target = target[:, gene_idx] if target is not None else None
    gene_ids_trace = gene_ids_trace[gene_idx]
    gene_names = np.asarray(gene_names)[gene_idx.detach().cpu().numpy()]
    return source, target, gene_ids_trace, gene_names

def make_initial_noise_like(source, seed=None):
    if config.noise_type == "Gaussian":
        if seed is None:
            return torch.randn_like(source)
        generator = torch.Generator(device=source.device)
        generator.manual_seed(int(seed))
        return torch.randn(
            source.shape,
            generator=generator,
            device=source.device,
            dtype=source.dtype,
        )
    if config.noise_type == "Poisson":
        if seed is None:
            return make_lognorm_poisson_noise(
                target_log=source,
                alpha=getattr(config, "poisson_alpha", 0.8),
                per_cell_L=getattr(config, "poisson_target_sum", 1e4),
            )
        cuda_devices = [source.device.index] if source.device.type == "cuda" and source.device.index is not None else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            return make_lognorm_poisson_noise(
                target_log=source,
                alpha=getattr(config, "poisson_alpha", 0.8),
                per_cell_L=getattr(config, "poisson_target_sum", 1e4),
            )
    raise ValueError(f"Unsupported noise_type: {config.noise_type}")

def train_step(source, target, perturbation_id, vf, criterion, accelerator, noise_type='Poisson', mode="predict_y"):
    B = source.shape[0]
    device = accelerator.device
    
    input_gene_ids = torch.randperm(source.shape[-1], device=device)[:config.infer_top_gene]
    source = source[:,input_gene_ids]
    target = target[:,input_gene_ids]
    gene = gene_ids.repeat(B,1).to(device)
    gene_input = gene[:,input_gene_ids]
    
    if mode=="predict_y":
        # source, target = ot_sampler.sample_plan(source, target)
        t = torch.rand(B, device=device)
        if noise_type=="Gaussian":
            target_noise = torch.randn_like(source)
        elif noise_type=="Poisson":
            target_noise = make_lognorm_poisson_noise(
                target_log=source,
                alpha=getattr(config, "poisson_alpha", 0.8),           
                per_cell_L=getattr(config, "poisson_target_sum", 1e4),  # e.g., 1e4 or None
            )
        path_x1 = path.sample(t=t, x_0=target_noise, x_1=target)
        predicted_x_t_velocity = vf(gene_input,path_x1.x_t, path_x1.t,source,perturbation_id, gene_input, mode=mode)
        loss = ((predicted_x_t_velocity - path_x1.dx_t)**2).mean()
        
        if config.use_mmd_loss:
            x1_hat = path_x1.x_t + predicted_x_t_velocity*(1-t).unsqueeze(-1)
            sigmas = median_sigmas(target, scales=(0.5,1.0,2.0,4.0))
            
            _mmd_loss = mmd2_unbiased_multi_sigma(x1_hat, target, sigmas)
            # _mmd_loss = mmd_loss(x1_hat, target)
            loss = loss + _mmd_loss * config.gamma

    elif mode=="predict_p":
        t_p = torch.ones(B, device=device)  # Or uniform(0.7,1.0)
        predicted_p_embed = vf(gene_input, target, t_p, source, perturbation_id, gene_input, mode=mode)
        if hasattr(vf, "module"):
            base_vf = vf.module
        else:
            base_vf = vf
        p_embed_gt = base_vf.get_perturbation_emb(perturbation_id=perturbation_id, cell_1=source)
        pred = F.normalize(predicted_p_embed, dim=-1)
        tgt  = F.normalize(p_embed_gt.detach(), dim=-1)
        loss = 1 - (pred * tgt).sum(dim=-1).mean()  # cosine distance
    
    return loss

@torch.inference_mode()
def test(data_sampler, vf, accelerator,  batch_size=128, path='./',vocab=None,scheme='mse'):
    gene_ids_test = vocab.encode(list(data_sampler.adata.var_names))
    
    gene_ids_test = torch.tensor(gene_ids_test, dtype=torch.long, device=device)
    perturbation_name_list = data_sampler._perturbation_covariates
    control_data = data_sampler.get_control_data()
    all_pred_expressions = [control_data['src_cell_data']]
    obs_perturbation_name_pred = ['control']*control_data['src_cell_data'].shape[0]
    all_target_expressions = [control_data['src_cell_data']]
    obs_perturbation_name_real = ['control']*control_data['src_cell_data'].shape[0]
    count = 0
    print('perturbation_name_list:',len(perturbation_name_list))
    for perturbation_name in perturbation_name_list:
        perturbation_data = data_sampler.get_perturbation_data(perturbation_name)
        target = perturbation_data['tgt_cell_data']
        perturbation_id = perturbation_data['condition_id']
        source = control_data['src_cell_data']
        source = source.to(device)
        perturbation_id = perturbation_id.to(device)
        if config.perturbation_function == 'crisper':
            perturbation_name_crisper = [inverse_dict[int(p_id)] for p_id in perturbation_id[0].cpu().numpy()]
            perturbation_id = torch.tensor(vocab.encode(perturbation_name_crisper), dtype=torch.long, device=device)
            perturbation_id = perturbation_id.repeat(source.shape[0],1)
        
        target_n = target.shape[0]
        eval_n_cells = getattr(config, "eval_n_cells", 128)
        N = target_n if eval_n_cells < 0 else min(eval_n_cells, target_n)
        if N <= source.shape[0]:
            idx = torch.randperm(source.shape[0], device=source.device)[:N]
        else:
            idx = torch.randint(source.shape[0], (N,), device=source.device)
        source = source[idx]
        
        pred_expressions = []
        for i in trange(0, N, batch_size):
            batch_perturbation_id = perturbation_id[0].repeat(source[i:i+batch_size].shape[0],1)
            
            batch_perturbation_id = batch_perturbation_id.to(accelerator.device)
            
            pred_expression = generate_sample(wrapped_vf,source[i:i+batch_size],batch_perturbation_id,vf,gene_ids=gene_ids_test,gene_all=gene_ids_test)
            pred_expressions.append(pred_expression)
            
        pred_expressions = torch.cat(pred_expressions, dim=0).cpu().numpy()
        all_pred_expressions.append(pred_expressions)
        all_target_expressions.append(target)
        obs_perturbation_name_pred.extend([perturbation_name] * pred_expressions.shape[0])
        obs_perturbation_name_real.extend([perturbation_name] * target.shape[0])
        # count += 1
        # if count > 3:
        #     break

    all_pred_expressions = np.concatenate(all_pred_expressions, axis=0)
    all_target_expressions = np.concatenate(all_target_expressions, axis=0)
    obs_pred = pd.DataFrame({'perturbation':obs_perturbation_name_pred})
    obs_real = pd.DataFrame({'perturbation':obs_perturbation_name_real})
    pred = ad.AnnData(X=all_pred_expressions, obs=obs_pred)
    real = ad.AnnData(X=all_target_expressions, obs=obs_real)
    

    eval_score = None
    if accelerator.is_main_process:
        pred.write_h5ad(os.path.join(path, 'pred.h5ad'))
        real.write_h5ad(os.path.join(path, 'real.h5ad'))

        evaluator = MetricsEvaluator(
            adata_pred=pred,
            adata_real=real,
            control_pert="control",
            pert_col="perturbation",
            num_threads=32,
        )
        (results, agg_results) = evaluator.compute()
        
        results.write_csv(os.path.join(path, 'results.csv'))
        agg_results.write_csv(os.path.join(path, 'agg_results.csv'))

        eval_score = pick_eval_score(agg_results, scheme)
        print(f"Current evaluation score: {eval_score:.4f}")
    
    return eval_score

def wrapped_vf(target,t,source,perturbation_id,vf,gene_ids, gene_all):
    
    gene = gene_ids.repeat(source.shape[0],1).to(device)
    predicted_x_t_velocity = vf(gene,target,t,source,perturbation_id,gene_all)
    
    return predicted_x_t_velocity

@torch.no_grad()
def generate_sample(wrapped_vf,source,condition_vec=None,vf=None,gene_ids=None,gene_all=None,steps=20,method="rk4"):
    target_noise = make_initial_noise_like(source)
        
    traj = torchdiffeq.odeint(lambda t,x: wrapped_vf(x,t,source,condition_vec,vf,gene_ids,gene_all),
                              target_noise,
                              torch.linspace(0,1,steps).to(source.device),
                              atol=1e-4,
                              rtol=1e-4,
                              method=method)
    # t = torch.linspace(0,1,steps).to(source.device)
    # traj = [target_noise + 0.8*wrapped_vf(target_noise,t,source,condition_vec,vf,gene_ids,gene_all)]
    
    return torch.clamp(traj[-1], min=0)

@torch.no_grad()
def generate_trace(source, condition_vec=None, vf=None, gene_ids=None, gene_all=None,
                   trace_times=None, initial_noise=None, method="rk4"):
    if trace_times is None:
        trace_times = parse_trace_times(config.trace_times)
    if initial_noise is None:
        initial_noise = make_initial_noise_like(source)

    ode_times = sorted(set([0.0, 1.0] + [float(t) for t in trace_times]))
    ode_times_tensor = torch.tensor(ode_times, device=source.device, dtype=source.dtype)
    traj = torchdiffeq.odeint(
        lambda t, x: wrapped_vf(x, t, source, condition_vec, vf, gene_ids, gene_all),
        initial_noise,
        ode_times_tensor,
        atol=1e-4,
        rtol=1e-4,
        method=method,
    )

    records = {}
    for trace_t in trace_times:
        trace_t = float(trace_t)
        trace_idx = min(range(len(ode_times)), key=lambda i: abs(ode_times[i] - trace_t))
        x_t = traj[trace_idx]
        t_tensor = torch.full((source.shape[0],), trace_t, device=source.device, dtype=source.dtype)
        velocity = wrapped_vf(x_t, t_tensor, source, condition_vec, vf, gene_ids, gene_all)
        x1_hat = x_t + (1.0 - trace_t) * velocity
        records[str(trace_t)] = {
            "x_t": x_t.detach(),
            "velocity": velocity.detach(),
            "x1_hat": x1_hat.detach(),
        }

    return records, torch.clamp(traj[-1], min=0)

def summarize_trace_tensor(tensor):
    return {
        "mean": tensor.mean(dim=0).detach().cpu().numpy(),
        "var": tensor.var(dim=0, unbiased=False).detach().cpu().numpy(),
    }

def adata_rows_to_tensor(adata, row_idx):
    x = adata.X[row_idx]
    if hasattr(x, "toarray"):
        x = x.toarray()
    return torch.from_numpy(np.asarray(x)).float()

@torch.inference_mode()
def export_flow_trace(data_sampler, vf, accelerator, path='./', vocab=None):
    if not accelerator.is_main_process:
        return

    trace_times = parse_trace_times(config.trace_times)
    trace_path = os.path.join(path, "trace")
    os.makedirs(trace_path, exist_ok=True)

    model = accelerator.unwrap_model(vf)
    was_training = model.training
    model.eval()

    gene_names_all = list(data_sampler.adata.var_names)
    gene_ids_trace = torch.tensor(vocab.encode(gene_names_all), dtype=torch.long, device=device)
    perturbation_name_list = list(data_sampler._perturbation_covariates)
    if config.trace_max_perturbations > 0:
        perturbation_name_list = perturbation_name_list[:config.trace_max_perturbations]

    groupby_obs = getattr(config, "trace_groupby_obs", "").strip()
    if groupby_obs:
        if groupby_obs not in data_sampler.adata.obs:
            raise ValueError(f"trace_groupby_obs={groupby_obs!r} not found in adata.obs")
        group_values = sorted(data_sampler.adata.obs[groupby_obs].astype(str).dropna().unique().tolist())
    else:
        group_values = ["pooled"]

    metadata = {
        "trace_times": trace_times,
        "trace_n_cells": int(config.trace_n_cells),
        "trace_n_seeds": int(config.trace_n_seeds),
        "trace_gene_panel": config.trace_gene_panel,
        "trace_save_cell_level": bool(config.trace_save_cell_level),
        "trace_groupby_obs": groupby_obs,
        "trace_groups": group_values,
        "noise_type": config.noise_type,
        "perturbations": perturbation_name_list,
    }
    with open(os.path.join(trace_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    obs = data_sampler.adata.obs
    if "is_control" in obs:
        control_mask = obs["is_control"].astype(bool).to_numpy()
    else:
        control_mask = (obs["perturbation_covariates"] == data_sampler.control_condition).to_numpy()

    for group_value in group_values:
        if groupby_obs:
            group_mask = (obs[groupby_obs].astype(str) == str(group_value)).to_numpy()
            group_label = str(group_value)
            group_path = os.path.join(trace_path, f"{sanitize_filename(groupby_obs)}={sanitize_filename(group_label)}")
        else:
            group_mask = np.ones(len(obs), dtype=bool)
            group_label = "pooled"
            group_path = trace_path

        source_idx_all = np.flatnonzero(control_mask & group_mask)
        if len(source_idx_all) == 0:
            print(f"skip trace group {group_label}: no control cells")
            continue

        source_all = adata_rows_to_tensor(data_sampler.adata, source_idx_all)
        n_trace_cells = min(config.trace_n_cells, source_all.shape[0])
        cell_generator = torch.Generator()
        cell_generator.manual_seed(0)
        cell_idx = torch.randperm(source_all.shape[0], generator=cell_generator)[:n_trace_cells]
        source_all = source_all[cell_idx].to(device)

        for perturbation_name in perturbation_name_list:
            target_mask = (obs["perturbation_covariates"] == perturbation_name).to_numpy() & group_mask
            target_idx = np.flatnonzero(target_mask)
            if len(target_idx) == 0:
                print(f"skip trace {group_label}/{perturbation_name}: no target cells")
                continue

            target_all = adata_rows_to_tensor(data_sampler.adata, target_idx).to(device)
            perturbation_id = torch.tensor(data_sampler.perturbation_covariates_id[target_idx], dtype=torch.long, device=device)

            if config.perturbation_function == 'crisper':
                perturbation_name_crisper = [inverse_dict[int(p_id)] for p_id in perturbation_id[0].cpu().numpy()]
                perturbation_id = torch.tensor(vocab.encode(perturbation_name_crisper), dtype=torch.long, device=device)
                perturbation_id = perturbation_id.repeat(source_all.shape[0], 1)
            else:
                perturbation_id = perturbation_id[0].repeat(source_all.shape[0], 1)

            source, target, gene_ids_panel, gene_names_panel = select_trace_gene_panel(
                source_all, target_all, gene_ids_trace, gene_names_all
            )
            perturbation_path = os.path.join(group_path, sanitize_filename(perturbation_name))
            os.makedirs(perturbation_path, exist_ok=True)

            source_summary = summarize_trace_tensor(source)
            target_summary = summarize_trace_tensor(target)

            for seed in range(config.trace_n_seeds):
                initial_noise = make_initial_noise_like(source, seed=seed)
                records, final = generate_trace(
                    source=source,
                    condition_vec=perturbation_id,
                    vf=model,
                    gene_ids=gene_ids_panel,
                    gene_all=gene_ids_panel,
                    trace_times=trace_times,
                    initial_noise=initial_noise,
                )

                x_t = torch.stack([records[str(float(t))]["x_t"] for t in trace_times], dim=0)
                velocity = torch.stack([records[str(float(t))]["velocity"] for t in trace_times], dim=0)
                x1_hat = torch.stack([records[str(float(t))]["x1_hat"] for t in trace_times], dim=0)

                payload = {
                    "perturbation": np.array(str(perturbation_name)),
                    "cell_line": np.array(str(group_label)),
                    "trace_groupby_obs": np.array(str(groupby_obs)),
                    "seed": np.array(seed),
                    "times": np.array(trace_times, dtype=np.float32),
                    "gene_names": np.asarray(gene_names_panel).astype(str),
                    "source_mean": source_summary["mean"],
                    "source_var": source_summary["var"],
                    "target_mean": target_summary["mean"],
                    "target_var": target_summary["var"],
                    "x_t_mean": x_t.mean(dim=1).detach().cpu().numpy(),
                    "x_t_var": x_t.var(dim=1, unbiased=False).detach().cpu().numpy(),
                    "velocity_mean": velocity.mean(dim=1).detach().cpu().numpy(),
                    "velocity_abs_mean": velocity.abs().mean(dim=1).detach().cpu().numpy(),
                    "x1_hat_mean": x1_hat.mean(dim=1).detach().cpu().numpy(),
                    "x1_hat_var": x1_hat.var(dim=1, unbiased=False).detach().cpu().numpy(),
                    "final_mean": final.mean(dim=0).detach().cpu().numpy(),
                    "final_var": final.var(dim=0, unbiased=False).detach().cpu().numpy(),
                }
                if config.trace_save_cell_level:
                    payload.update({
                        "x_t": x_t.detach().cpu().numpy(),
                        "velocity": velocity.detach().cpu().numpy(),
                        "x1_hat": x1_hat.detach().cpu().numpy(),
                        "final": final.detach().cpu().numpy(),
                    })

                out_file = os.path.join(perturbation_path, f"seed_{seed}.npz")
                np.savez_compressed(out_file, **payload)
                print(f"saved flow trace: {out_file}")

    if was_training:
        model.train()
    
if __name__ == "__main__":
    config = tyro.cli(Config)

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))

    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs, init_kwargs]
    )
    if accelerator.is_main_process:
        print(config)
        save_path = config.make_path()
        os.makedirs(save_path, exist_ok=True)
    device = accelerator.device
    if torch.cuda.is_available():
        torch.cuda.set_device(accelerator.local_process_index)
    
    data_manager = Data('./data')
    data_manager.data_name = config.data_name

    is_known_dataset = config.data_name in ['norman', 'norman_umi_go_filtered', 'combosciplex']
    if is_known_dataset:
        data_manager.load_data(config.data_name)
        data_manager.process_data(n_top_genes=config.n_top_genes, infer_top_gene=config.infer_top_gene, split_method=config.split_method, fold=config.fold, use_negative_edge=config.use_negative_edge, k=config.topk, mask_max_cells=config.mask_max_cells)
    else:
        if accelerator.is_main_process:
            data_manager.load_data(config.data_name)
            data_manager.process_data(n_top_genes=config.n_top_genes, infer_top_gene=config.infer_top_gene, split_method=config.split_method, fold=config.fold, use_negative_edge=config.use_negative_edge, k=config.topk, mask_max_cells=config.mask_max_cells)
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            data_manager.process_data(n_top_genes=config.n_top_genes, infer_top_gene=config.infer_top_gene, split_method=config.split_method, fold=config.fold, use_negative_edge=config.use_negative_edge, k=config.topk, mask_max_cells=config.mask_max_cells)
        accelerator.wait_for_everyone()
    train_sampler, valid_sampler, test_dl = data_manager.load_flow_data(batch_size=config.batch_size)
    
    train_dataset = PerturbationDataset(train_sampler, config.batch_size)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True)  # batch_size=1 因为每个getitem本身就是一个batch
    if config.use_negative_edge:
        mask_path = os.path.join(data_manager.data_path, data_manager.data_name,'mask_fold_'+str(config.fold)+'topk_'+str(config.topk)+config.split_method+'_negative_edge'+'.pt')
    else:
        mask_path = os.path.join(data_manager.data_path, data_manager.data_name,'mask_fold_'+str(config.fold)+'topk_'+str(config.topk)+config.split_method+'.pt')
    vocab = process_vocab(data_manager, config)

    gene_ids = vocab.encode(list(data_manager.adata.var_names))
    perturbation_ntoken = 0
    if config.perturbation_function != 'crisper' and hasattr(data_manager, "perturbation_dict"):
        perturbation_ntoken = max(data_manager.perturbation_dict.values(), default=-1) + 1
    model_ntoken = max(config.ntoken, len(vocab), perturbation_ntoken)
    if model_ntoken != config.ntoken and accelerator.is_main_process:
        print(
            f"##### expanding model ntoken from {config.ntoken} to {model_ntoken} "
            f"(vocab={len(vocab)}, perturbations={perturbation_ntoken}) #####"
        )
    
    vf = instantiate_model(config.model_type,
                           ntoken = model_ntoken,
                           d_model = config.d_model,
                           d_perturbation = config.d_model,
                           fusion_method = config.fusion_method,
                           perturbation_function = config.perturbation_function,
                           mask_path = mask_path
                           )
    
    model_path = config.make_path()
    
    gene_ids = torch.tensor(gene_ids, dtype=torch.long, device=device)
    
    save_path = config.make_path()
    best_loss = float('inf')
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(vf.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps, eta_min=config.eta_min)
    
    if config.checkpoint_path != '':
        _, _ = load_checkpoint(config.checkpoint_path, vf, optimizer, scheduler)
    start_iteration = 0 
    vf = accelerator.prepare(vf)
    optimizer, scheduler, dataloader = accelerator.prepare(optimizer,scheduler,dataloader)
    inverse_dict = {v: str(k) for k, v in data_manager.perturbation_dict.items()}

    if config.test_only:
        inference_path = os.path.join(save_path, "test_only")
        if accelerator.is_main_process:
            os.makedirs(inference_path, exist_ok=True)
        accelerator.wait_for_everyone()

        if config.eval_every != 0:
            test(valid_sampler, vf, accelerator, batch_size=config.batch_size, path=inference_path, vocab=vocab)
        if config.trace_every > 0:
            export_flow_trace(valid_sampler, vf, accelerator, path=inference_path, vocab=vocab)

        accelerator.wait_for_everyone()
        raise SystemExit

    tb_writer = None
    if config.use_tensorboard and accelerator.is_main_process:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoard monitoring requires tensorboard. "
                "Install it in the training environment or run without --use_tensorboard."
            ) from exc

        tb_log_dir = config.tensorboard_log_dir or os.path.join(save_path, "tensorboard")
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard log dir: {tb_log_dir}")

    pbar = tqdm.tqdm(total=config.steps, initial=start_iteration)
    iteration = start_iteration
    while iteration < config.steps:
        for batch_data in dataloader:
            if iteration >= config.steps:
                break
            
            source = batch_data['src_cell_data'].squeeze(0)
            target = batch_data['tgt_cell_data'].squeeze(0)
            perturbation_id = batch_data['condition_id'].squeeze(0).to(device)
            if config.perturbation_function == 'crisper':
                perturbation_name = [inverse_dict[int(p_id)] for p_id in perturbation_id[0].cpu().numpy()]
                perturbation_id = torch.tensor(vocab.encode(perturbation_name), dtype=torch.long, device=device)
                perturbation_id = perturbation_id.repeat(source.shape[0],1)
            
            
            set_requires_grad_for_p_only(vf, p_only=config.mode)
            loss = train_step(source, target, perturbation_id, vf, criterion, accelerator, noise_type=config.noise_type, mode=config.mode)
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            loss_for_log = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
            lr_for_log = scheduler.get_last_lr()[0]
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", loss_for_log, iteration)
                tb_writer.add_scalar("train/lr", lr_for_log, iteration)

            
            if iteration % config.print_every == 0:
                save_path_ = os.path.join(save_path, f'iteration_{iteration}')
                os.makedirs(save_path_, exist_ok=True)
                if accelerator.is_main_process:
                    print(f"svaing {iteration}'s checkpoint...")
                    
                    save_checkpoint(
                        model=accelerator.unwrap_model(vf), 
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        iteration=iteration, 
                        eval_score=None,  # 不需要评估分数
                        save_path=save_path_, 
                        is_best=False
                    )
                should_eval = (
                    iteration != 0
                    and (
                        (config.eval_every < 0)
                        or (config.eval_every > 0 and iteration % config.eval_every == 0)
                    )
                )
                if not should_eval:
                    if accelerator.is_main_process:
                        print(f"skip iteration {iteration} full evaluation")
                else:
                    eval_score = test(valid_sampler, vf, accelerator, batch_size=config.batch_size, path=save_path_,vocab=vocab)

                should_trace = (
                    config.trace_every > 0
                    and iteration != 0
                    and iteration % config.trace_every == 0
                )
                if should_trace:
                    export_flow_trace(valid_sampler, vf, accelerator, path=save_path_, vocab=vocab)
                
            accelerator.wait_for_everyone()
            
            pbar.update(1)
            pbar.set_description(f'loss: {loss_for_log:.4f}, iteration: {iteration}')
            iteration += 1

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
            
