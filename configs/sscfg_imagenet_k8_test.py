# name = 'sscfg_imagenet_k8_test'
name = 'separate_gm_imagenet_k8_test'

model = dict(
    type='LatentDiffusionClassImage',
    vae=dict(
        type='PretrainedVAE',
        from_pretrained='stabilityai/sd-vae-ft-ema',
        freeze=True,
        torch_dtype='bfloat16'),
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
            torch_dtype='bfloat16'),
        spectrum_net=dict(
            type='SpectrumMLP',
            base_size=(4, 32, 32),
            layers=[64, 8],
            torch_dtype='bfloat16'),
        num_timesteps=1000,
        timestep_sampler=dict(type='ContinuousTimeStepSampler', shift=1.0, logit_normal_enable=True),
        denoising_mean_mode='U'),
    diffusion_use_ema=True,
    inference_only=True)

save_interval = 1000
must_save_interval = 20000  # interval to save regardless of max_keep_ckpts
eval_interval = 10000
work_dir = 'work_dirs/' + name

train_cfg = dict()
test_cfg = dict(
    latent_size=(4, 32, 32),
    single_step_cfg=True,
    # T_low=0.4,
    # T_high=3,
)

optimizer = {}
data = dict(
    workers_per_gpu=4,
    val=dict(
        num_test_images=1600, # TODO: change to 50_000 for final eval
        num_test_images=1600, # TODO: change to 50_000 for final eval
        type='ImageNet',
        test_mode=True),
    val_dataloader=dict(samples_per_gpu=80),
    test_dataloader=dict(samples_per_gpu=80),
    val_dataloader=dict(samples_per_gpu=80),
    test_dataloader=dict(samples_per_gpu=80),
    persistent_workers=True,
    prefetch_factor=64)
lr_config = dict()
checkpoint_config = dict()


guidance_scales = [2]
# temperatures = [0.8, 0.7, 0.6]

methods = dict(
    gmode1=dict(
        output_mode='mean',
        sampler='FlowEulerODE',
        order=1),
    # gmsde2=dict(
    #     output_mode='sample',
    #     sampler='GMFlowSDE',
    #     order=2)
)

evaluation = []


for step, substep in [(8, 1)]:  # (8, 16)
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


# for step, substep in [(8, 1), (32, 1)]:
#     for method_name, method_config in methods.items():
#         for guidance_scale in guidance_scales:
#             for temp in temperatures:
#                 temp_str = str(temp) # f"{t_low:.2f}-{t_high:.2f}"
#                 test_cfg_override = dict(
#                     orthogonal_guidance=1,
#                     guidance_scale=guidance_scale,
#                     temperature=temp,
#                     # T_low=t_low,
#                     # T_high=t_high,
#                     num_timesteps=step)
#                 if 'gmode' in method_name:
#                     test_cfg_override.update(num_substeps=substep)
#                 test_cfg_override.update(method_config)
#                 evaluation.append(
#                     dict(
#                         type='GenerativeEvalHook',
#                         data='val',
#                         prefix=f'{method_name}_g{guidance_scale:.2f}_temp{temp_str}_step{step}',
#                         sample_kwargs=dict(
#                             test_cfg_override=test_cfg_override),
#                         interval=eval_interval,
#                         feed_batch_size=32,
#                         viz_step=512,
#                         metrics=[
#                             dict(
#                                 type='InceptionMetrics',
#                                 resize=False,
#                                 reference_pkl='huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl'),
#                         ],
#                         viz_dir='viz/' + name + f'/{method_name}_g{guidance_scale:.2f}_temp{temp_str}_step{step}',
#                         save_best_ckpt=False))

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


# python test.py configs/sscfg_imagenet_k8_test.py /workspace/GMFlow/checkpoints/gmflow_imagenet_k8_8gpus/latest.pth --deterministic --gpu-ids 0
