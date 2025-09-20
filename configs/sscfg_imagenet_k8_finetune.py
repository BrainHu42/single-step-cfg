# 512 samples per gpu, requires 40GB VRAM
name = 'sscfg_imagenet_k8_finetune'

model = dict(
    type='LatentDiffusionClassImage',
    vae=dict(
        type='PretrainedVAE',
        from_pretrained='stabilityai/sd-vae-ft-ema',
        freeze=True,
        torch_dtype='float16'),
    diffusion=dict(
        type='GMFlowSSCFG',
        denoising=dict(
            type='GMDiTTransformer2DModel',
            num_gaussians=8,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            num_attention_heads=16,
            attention_head_dim=72,
            in_channels=4,
            num_layers=28,
            sample_size=32,  # 256
            torch_dtype='float32',
            pretrained='/workspace/gmflow_imagenet_k8_ema/transformer/diffusion_pytorch_model.bf16.pt',
            checkpointing=True),
        spectrum_net=dict(
            type='SpectrumMLP',
            base_size=(4, 32, 32),
            layers=[64, 8],
            pretrained='/workspace/gmflow_imagenet_k8_ema/spectrum_net/diffusion_pytorch_model.bf16.pt',
            torch_dtype='float32'),
        flow_loss=dict(
            type='GMFlowHybridLoss',
            log_cfgs=dict(type='quartile', prefix_name='loss_trans', total_timesteps=1000),
            data_info=None,
            p_uncond=0.25,
            # top_k=16,
            weight_scale=2.0),  # I'm not using _forward_loss
        num_timesteps=1000,
        timestep_sampler=dict(type='ContinuousTimeStepSampler', shift=1.0, logit_normal_enable=True),
        denoising_mean_mode='U'),
    diffusion_use_ema=True,
    autocast_dtype='bfloat16')

save_interval = 1000
must_save_interval = 40000  # interval to save regardless of max_keep_ckpts
eval_interval = 20000
work_dir = 'work_dirs/' + name
total_iters = 100_000 

train_cfg = dict(
    prob_class=1.0,
    trans_ratio=0.5,
    diffusion_grad_clip=10.0,
    diffusion_grad_clip_begin_iter=1000,
    grad_accum_steps=5,
)
test_cfg = dict(
    latent_size=(4, 32, 32),
)

optimizer = {
    'diffusion': dict(
        type='AdamW8bit', lr=2e-5, betas=(0.9, 0.95), weight_decay=0.05,  # old_lr / 10
        paramwise_cfg=dict(custom_keys={
            'bias': dict(decay_mult=0.0),
        })
    ),
}
data = dict(
    workers_per_gpu=4,
    train=dict(
        type='ImageNet',
        data_root='data/imagenet/train_cache',
        datalist_path='data/imagenet/train_cache.txt'),
    train_dataloader=dict(samples_per_gpu=128),
    val=dict(
        num_test_images=10_000,
        type='ImageNet',
        test_mode=True),
    val_dataloader=dict(samples_per_gpu=125),
    test_dataloader=dict(samples_per_gpu=125),
    persistent_workers=True,
    prefetch_factor=256)
lr_config = dict(
    policy='fixed',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001)
checkpoint_config = dict(
    interval=save_interval,
    must_save_interval=must_save_interval,
    by_epoch=False,
    max_keep_ckpts=1,
    out_dir='checkpoints/')

step = 16
substep = 8
guidance_scale = 0.1

evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=f'gmode2_g{guidance_scale:.2f}_step{step}',
        sample_kwargs=dict(
            test_cfg_override=dict(
                output_mode='mean',
                sampler='FlowEulerODE',
                guidance_scale=guidance_scale,
                order=2,
                num_timesteps=step,
                num_substeps=substep,
            )),
        interval=eval_interval,
        feed_batch_size=32,
        viz_step=256,
        metrics=[
            dict(
                type='InceptionMetrics',
                resize=False,
                reference_pkl='huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl'),
            ],
        viz_dir='viz/' + name + f'/gmode2_g{guidance_scale:.2f}_step{step}',
        save_best_ckpt=False)]

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHookMod',
        module_keys=('diffusion_ema', ),
        interp_mode='lerp',
        interval=1,
        start_iter=0,
        momentum_policy='rampup',
        momentum_cfg=dict(
            ema_kimg=30000, ema_rampup=0.05, batch_size=4096, eps=1e-8),
        priority='VERY_HIGH'),
]

# use dynamic runner
runner = dict(
    type='DynamicIterBasedRunnerMod',
    is_dynamic_ddp=False,
    pass_training_status=True,
    ckpt_trainable_only=True,
    ckpt_fp16_ema=True)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = f'checkpoints/{name}/latest.pth'  # resume by default
workflow = [('train', save_interval)]
use_ddp_wrapper = True
find_unused_parameters = False
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'


# OG 4090: 2025-09-07 07:30:20,226 - mmgen - INFO - Iter [3000/200000]     lr_diffusion: 2.000e-04, eta: 3 days, 9:18:23, time: 2.034, data_time: 0.006, memory: 14763, loss_trans_quartile_0: -5.6522, loss_trans_var_quartile_0: 0.0275, loss_trans_quartile_1: -2.7076, loss_trans_var_quartile_1: 0.1059, loss_trans_quartile_2: -1.1508, loss_trans_var_quartile_2: 0.2209, loss_trans_quartile_3: -0.3157, loss_trans_var_quartile_3: 0.3257, loss_transition: -4.4789, loss_spectral: -0.0922, diffusion_grad_norm: 0.1976

# 1x 4090: 2025-09-10 19:14:51,612 - mmgen - INFO - Iter [6520/100000]     lr_diffusion: 1.000e-04, eta: 1 day, 12:20:27, time: 1.388, data_time: 0.005, memory: 16629, loss_cond: 2.0930, loss_uncond: 0.6376, loss_diffusion: 1.8019, loss_spectral: -0.1289, diffusion_grad_norm: 0.1383
# 1x 5090: 2025-09-11 17:42:58,247 - mmgen - INFO - Iter [18160/100000]    lr_diffusion: 1.000e-04, eta: 22:38:09, time: 0.972, data_time: 0.004, memory: 12081, loss_cond: 2.1791, loss_uncond: 0.6116, loss_diffusion: 1.8656, loss_spectral: -0.1301, diffusion_grad_norm: 0.1374