# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # We added a WandbLoggerHook to collect all the metrics
        # It will be modified in tools/train.py
        dict(type='WandbLoggerHook', init_kwargs=dict(project='testing', name='test')),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
