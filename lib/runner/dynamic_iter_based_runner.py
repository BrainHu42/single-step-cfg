import os.path as osp
import platform
import shutil
import torch
import mmcv

from torch.optim import Optimizer
from mmcv.runner import RUNNERS
from mmgen.core import DynamicIterBasedRunner

from .checkpoint import save_checkpoint


@RUNNERS.register_module()
class DynamicIterBasedRunnerMod(DynamicIterBasedRunner):
    
    def __init__(self, *args, ckpt_trainable_only=False, ckpt_fp16_ema=False, **kwargs):
        super(DynamicIterBasedRunnerMod, self).__init__(*args, **kwargs)
        self.ckpt_trainable_only = ckpt_trainable_only
        self.ckpt_fp16_ema = ckpt_fp16_ema

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='iter_{}.pth',
                        meta=None,
                        save_optimizer=True,
                        create_symlink=True):
        if meta is None:
            meta = dict(iter=self.iter + 1, epoch=self.epoch + 1)
        elif isinstance(meta, dict):
            meta.update(iter=self.iter + 1, epoch=self.epoch + 1)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        _loss_scaler = self.loss_scaler if self.with_fp16_grad_scaler else None
        save_checkpoint(
            self.model,
            filepath,
            optimizer=optimizer,
            loss_scaler=_loss_scaler,
            save_apex_amp=self.use_apex_amp,
            meta=meta,
            trainable_only=self.ckpt_trainable_only,
            fp16_ema=self.ckpt_fp16_ema)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               resume_loss_scaler=True,
               map_location='default'):
        try:
            if map_location == 'default':
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(
                    checkpoint, map_location=map_location)
        except Exception as e:
            self.logger.warning(f'load checkpoint {checkpoint} failed: {e}')
            return False

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        self._inner_iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            optimizer_state = checkpoint['optimizer']
            if isinstance(self.optimizer, Optimizer):
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                except (ValueError, KeyError, RuntimeError) as err:
                    self.logger.warning(
                        'optimizer state_dict mismatch detected during resume: '
                        f'{err}. Proceeding without loading optimizer state.')
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    if k not in optimizer_state:
                        self.logger.warning(
                            f'skip loading optimizer state for "{k}"; '
                            'entry missing in checkpoint.')
                        continue
                    try:
                        self.optimizer[k].load_state_dict(optimizer_state[k])
                    except (ValueError, KeyError, RuntimeError) as err:
                        self.logger.warning(
                            'optimizer state_dict mismatch detected for '
                            f'"{k}" during resume: {err}. '
                            'Proceeding without loading this optimizer state.')
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        if 'loss_scaler' in checkpoint and resume_loss_scaler and hasattr(self, 'loss_scaler'):
            self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])

        if self.use_apex_amp:
            from apex import amp
            amp.load_state_dict(checkpoint['amp'])

        self.logger.info(f'resumed from epoch: {self.epoch}, iter {self.iter}')
