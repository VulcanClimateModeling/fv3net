from ml_collections import config_dict

config = config_dict.ConfigDict()

config.dim = 64
config.dim_mults = (1, 1, 2, 2, 3, 4)
config.learned_sinusoidal_cond = True,
config.random_fourier_features = True,
config.learned_sinusoidal_dim = 32
config.diffusion_steps = 1500
config.sampling_steps = 20
config.loss = "l2"
config.objective = "pred_v"
config.lr = 1e-4
config.steps = 700000
config.grad_acc = 1
config.val_num_of_batch = 5
config.save_and_sample_every = 20000
config.ema_decay = 0.995
config.amp = False
config.split_batches = True
config.additional_note = "2d-nomulti-nols-ensemble"
config.eval_folder = "./evaluate"
config.results_folder = "./results"
config.tensorboard_dir = "./tensorboard"
config.milestone = 2
config.rollout = "partial"
config.rollout_batch = 25

config.batch_size = 2
config.data_config = config_dict.ConfigDict({
    "dataset_name": "c384",
    "length": 7,
    "channels": ["PRATEsfc_coarse"],
    "img_channel": 1,
    "img_size": 384,
    "logscale": False,
    "multi": False,
    "flow": "2d",
    "minipatch": False
})

config.data_name = f"{config.data_config['dataset_name']}-{config.data_config['channels']}-{config.objective}-{config.loss}-d{config.dim}-t{config.diffusion_steps}{config.additional_note}"
config.model_name = f"c384-{config.data_config['channels']}-{config.objective}-{config.loss}-d{config.dim}-t{config.diffusion_steps}{config.additional_note}"