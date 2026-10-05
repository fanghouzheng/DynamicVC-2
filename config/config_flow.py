
from dataclasses import dataclass
import os
@dataclass
class FlowConfig:
    # Flow model type
    model_type: str = 'hierarchical'

    # Flow Matching specific parameters
    batch_size: int = 32
    ntoken: int = 512
    d_model: int = 512
    lr: float = 1e-5
    steps: int = 5000
    eta_min: float = 1e-7
    devices: str = "1"
    test_only: bool = False
    # Perturbation related parameters
    data_name: str = "combosciplex"
    perturbation_function: str = 'crisper' 
    noise_type: str = "Gaussian"
    poisson_alpha: float = 0.8
    poisson_target_sum: int = -1

    print_every: int = 5000
    mode: str = 'predict_y' # predict_y, predict_p
    result_path: str = './result'
    perturbation_fusion_method: str = 'sum' # mlp, sum
    fusion_method: str = 'cross' # cross , concat, add
    infer_top_gene: int = 1000
    n_top_genes: int = 5000
    checkpoint_path: str = ''
    gamma: float = 0.0
    split_method: str = 'additive'
    use_mmd_loss: bool = False
    fold: int = 0
    use_negative_edge: bool = False
    topk: int = 15
    mask_max_cells: int = -1  # >0: sample this many train cells when building gene coexpression mask
    eval_every: int = -1  # -1: evaluate with print_every; 0: disable eval; >0: evaluate every N iterations
    eval_n_cells: int = 128  # >0: predicted cells per perturbation; -1: match real target cells

    # Flow trace export for structured SimContext construction.
    # Disabled by default; set trace_every > 0 to export after matching checkpoints.
    trace_times: str = "0.2,0.5,0.8,1.0"
    trace_n_cells: int = 128
    trace_n_seeds: int = 5
    trace_every: int = 0
    trace_max_perturbations: int = -1
    trace_gene_panel: str = "all"  # all, infer_top_gene
    trace_save_cell_level: bool = False
    trace_groupby_obs: str = ""  # e.g. cell_name; empty means pooled controls/targets

    # TensorBoard monitoring.
    use_tensorboard: bool = False
    tensorboard_log_dir: str = ""  # empty: <experiment_save_path>/tensorboard
    
    def __post_init__(self):
        if self.data_name == 'norman_umi_go_filtered':
            self.n_top_genes = 5054
        if self.data_name == 'norman':
            self.n_top_genes = 5000
        path = self.make_path()

    def make_path(self):
        exp_name = '-'.join(['flow', 
                             f'fusion_{self.fusion_method}',
                            f'{self.data_name}', 
                            self.model_type, 
                            self.mode, 
                            f'gamma_{self.gamma}',
                            f'perturbation_function_{self.perturbation_function}',
                            f'lr_{self.lr}', 
                            f'dim_model_{self.d_model}', 
                            f'infer_top_gene_{self.infer_top_gene}',
                            f'split_method_{self.split_method}',
                            f'use_mmd_loss_{self.use_mmd_loss}',
                            f'fold_{self.fold}',
                            f'use_negative_edge_{self.use_negative_edge}',
                            f'topk_{self.topk}',
                            ])
        return os.path.join(self.result_path, exp_name)
