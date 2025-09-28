name = 'sscfg_fm_imagenet_k8_test'

model = dict(
    type='LatentDiffusionClassImage',
    vae=dict(
        type='PretrainedVAE',
        from_pretrained='stabilityai/sd-vae-ft-ema',
        freeze=True,
        torch_dtype='float16'),
    diffusion=dict(
        type='GaussianFlow',
        denoising=dict(
            type='GMDiTTransformer2DModel_Uncond',
            num_gaussians=1,
            constant_logstd=0.0,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=1,
            num_attention_heads=16,
            attention_head_dim=72,
            in_channels=4,
            num_layers=28,
            sample_size=32,  # 256
            torch_dtype='float32',
            checkpointing=True),
        flow_loss=dict(
            type='DDPMMSELossMod',
            p_uncond=0.4,
            # Let the loss compute u_t and uncond_u_t from mixture heads
            num_timesteps=1000,
            weight_scale=2.0),
        num_timesteps=1000,
        timestep_sampler=dict(type='ContinuousTimeStepSampler', shift=2.0, logit_normal_enable=True),
        denoising_mean_mode='U'),
    diffusion_use_ema=True,
    autocast_dtype='bfloat16')

save_interval = 1000
must_save_interval = 20000  # interval to save regardless of max_keep_ckpts
eval_interval = 10000
work_dir = 'work_dirs/' + name

train_cfg = dict()
test_cfg = dict(
    latent_size=(4, 32, 32),
    single_step_cfg=True,
)

optimizer = {}
data = dict(
    workers_per_gpu=4,
    val=dict(
        num_test_images=1600, # TODO: change to 50_000 for final eval
        type='ImageNet',
        test_mode=True),
    val_dataloader=dict(samples_per_gpu=80),
    test_dataloader=dict(samples_per_gpu=80),
    persistent_workers=True,
    prefetch_factor=64)
lr_config = dict()
checkpoint_config = dict()


guidance_scales = [2.2, 2.4, 2.6, 2.8, 3.0]

methods = dict(
    gmode2=dict(
        output_mode='mean',
        sampler='FlowEulerODE',
        order=1),
    # gmsde2=dict(
    #     output_mode='sample',
    #     sampler='GMFlowSDE',
    #     order=2)
)

evaluation = []

for step, substep in [(16, 1)]:  # (8, 16)
    for method_name, method_config in methods.items():
        for guidance_scale in guidance_scales:
            test_cfg_override = dict(
                orthogonal_guidance=1.0,
                guidance_scale=guidance_scale,
                num_timesteps=step)
            if 'gmode' in method_name:
                test_cfg_override.update(num_substeps=substep)
            test_cfg_override.update(method_config)
            evaluation.append(
                dict(
                    type='GenerativeEvalHook',
                    data='val',
                    prefix=f'{method_name}_g{guidance_scale:.2f}_step{step}',
                    sample_kwargs=dict(
                        test_cfg_override=test_cfg_override),
                    interval=eval_interval,
                    feed_batch_size=32,
                    viz_step=512,
                    metrics=[
                        dict(
                            type='InceptionMetrics',
                            resize=False,
                            reference_pkl='huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl'),
                    ],
                    viz_dir='viz/' + name + f'/{method_name}_g{guidance_scale:.2f}_step{step}',
                    save_best_ckpt=False))

total_iters = 200000
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

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
resume_from = None
workflow = [('train', save_interval)]
use_ddp_wrapper = True
find_unused_parameters = False
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'

# python test.py configs/sscfg_fm_imagenet_k8_test.py /data/single-step-cfg/checkpoints/sscfg_fm_imagenet_k8_8gpus/iter_97000.pth --gpu-ids 0