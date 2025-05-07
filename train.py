import argparse
import torch
import os
import mmcv
import logging
import time
import cv2
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import src.utils.solver_utils as solver_utils
import src.utils.common as comm
import os.path as osp
from train_utils import batch_data_self
from src.utils.load_model import load_model
from src.utils.build_optimizers import build_optimizer
from src.data.dataset_factory import register_datasets_in_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from src.data.data_loader import build_gdrn_self_train_loader, build_gdrn_test_loader
from src.utils.utils import dprint
from src.utils.my_checkpoint import MyCheckpointer
from src.utils.torch_utils.torch_utils import ModelEMA
from src.utils.torch_utils.misc import nan_to_num
from src.utils.my_writer import MyCommonMetricPrinter, MyJSONWriter, MyTensorboardXWriter
from src.utils.gdrn_custom_evaluator import GDRN_EvaluatorCustom
from src.utils.gdrn_evaluator import gdrn_inference_on_dataset
from train_utils import compute_self_loss
from datetime import timedelta
from collections import OrderedDict
from omegaconf import OmegaConf
from torch.cuda.amp import autocast, GradScaler
from detectron2.checkpoint import PeriodicCheckpointer
from detectron2.utils.events import EventStorage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_tbx_event_writer(out_dir, backup=False):
    tb_logdir = osp.join(out_dir, "tb")
    mmcv.mkdir_or_exist(tb_logdir)
    if backup and comm.is_main_process():
        old_tb_logdir = osp.join(out_dir, "tb_old")
        mmcv.mkdir_or_exist(old_tb_logdir)
        os.system("mv -v {} {}".format(osp.join(tb_logdir, "events.*"), old_tb_logdir))

    tbx_event_writer = MyTensorboardXWriter(tb_logdir, backend="tensorboardX")
    return tbx_event_writer

def train_worker(gpu_id: int, world_size: int, config: OmegaConf):
    torch.cuda.set_device(gpu_id)
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        timeout=timedelta(seconds=720000),
        rank=gpu_id,
        world_size=world_size,
    )

    # Prepare the dataset
    register_datasets_in_cfg(config)
    dataset_meta = MetadataCatalog.get(config.datasets.train[0])
    obj_names = dataset_meta.objs

    # Load data
    train_dset_names = config.datasets.train
    data_loader = build_gdrn_self_train_loader(config, train_dset_names, train_objs=obj_names)
    data_loader_iter = iter(data_loader)

    batch_size = config.train.batch_size
    dataset_len = len(data_loader.dataset)
    iters_per_epoch = dataset_len // batch_size // world_size
    max_iter = config.train.epochs * iters_per_epoch
    dprint("batch_size: ", batch_size)
    dprint("dataset length: ", dataset_len)
    dprint("iters per epoch: ", iters_per_epoch)
    dprint("total iters: ", max_iter)

    # Initialize the model
    model, model_teacher, params = load_model(config)
    model.cuda()
    model_teacher.cuda()
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[gpu_id],
        find_unused_parameters=True,
    )
    model_teacher = nn.parallel.DistributedDataParallel(
        model_teacher,
        device_ids=[gpu_id],
        find_unused_parameters=True,
    )                    
    model.train()
    model_teacher.eval()
    
    # Initialize the optimizer
    optimizer = build_optimizer(config, params)
    # Initialize the scheduler
    scheduler = solver_utils.build_lr_scheduler(config, optimizer, total_iters=max_iter)

    # Initialize the checkpointer
    grad_scaler = GradScaler()
    checkpointer = MyCheckpointer(
        model,
        save_dir=config.output_dir,
        model_teacher=model_teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        gradscaler=grad_scaler,
        save_to_disk=comm.is_main_process(),
    )
    # Load the checkpoint for student model
    start_iter = checkpointer.resume_or_load(config.model.pretrained, resume=False).get("iteration", -1) + 1
    start_epoch = start_iter // iters_per_epoch + 1  # first epoch is 1

    # Exponential moving average for teacher (NOTE: initialize ema after loading weights) ========================
    if comm.is_main_process():
        ema = ModelEMA(model_teacher, **config.model.ema.init_cfg)
        ema.updates = start_epoch // config.model.ema.update_freq
        # save the ema model
        checkpointer.model = ema.ema.module if hasattr(ema.ema, "module") else ema.ema
    else:
        ema = None

    if config.train.checkpoint_by_epoch:
        ckpt_period = config.train.checkpoint_period * iters_per_epoch
    else:
        ckpt_period = config.train.checkpoint_period
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, ckpt_period, max_iter=max_iter, max_to_keep=config.train.max_to_keep
    )
    
    # Build writers
    tbx_event_writer = get_tbx_event_writer(config.output_dir, backup=not config.get("RESUME", False))
    tbx_writer = tbx_event_writer._writer  # NOTE: we want to write some non-scalar data
    writers = (
        [MyCommonMetricPrinter(max_iter), MyJSONWriter(osp.join(config.output_dir, "metrics.json")), tbx_event_writer]
        if comm.is_main_process()
        else []
    )
    # test(config, model.module, epoch=0, iteration=0)
    logger.info("Starting training from iteration {}".format(start_iter))
    iter_time = None
    with EventStorage(start_iter) as storage:
        optimizer.zero_grad(set_to_none=True)
        for iteration in range(start_iter, max_iter):
            storage.iter = iteration
            epoch = iteration // iters_per_epoch + 1  # epoch start from 1
            storage.put_scalar("epoch", epoch, smoothing_hint=False)
            is_log_iter = False
            if iteration - start_iter > 5 and (
                (iteration + 1) % config.train.print_freq == 0 or iteration == max_iter - 1 or iteration < 100
            ):
                is_log_iter = True
            
            data = next(data_loader_iter)

            if iter_time is not None:
                storage.put_scalar("time", time.perf_counter() - iter_time)
            iter_time = time.perf_counter()

            # Forward
            file_names = []
            batch = batch_data_self(config, data, file_names, model_teacher=model_teacher)

            # only outputs, no losses
            out_dict = model(
                batch["roi_img"],
                gt_points=batch.get("roi_points", None),
                sym_infos=batch.get("sym_info", None),
                roi_classes=batch["roi_cls"],
                roi_cams=batch["roi_cam"],
                roi_whs=batch["roi_wh"],
                roi_centers=batch["roi_center"],
                resize_ratios=batch["resize_ratio"],
                roi_coord_2d=batch.get("roi_coord_2d", None),
                roi_extents=batch.get("roi_extent", None),
                do_self=True,
            )
            
            # Compute losses
            loss_dict = compute_self_loss(
                config,
                batch,
                out_dict["rot"],
                out_dict["trans"],
            )
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            # Backward
            losses.backward()
            # optimize
            # set nan grads to 0
            for param in model.parameters():
                if param.grad is not None:
                    nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
            optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            # EMA update
            if ema is not None and (iteration + 1) % (config.model.ema.update_freq * iters_per_epoch) == 0:
                ema.update(model)
                ema.update_attr(model)

            # Do test
            if config.test.eval_period > 0 and (iteration + 1) % config.test.eval_period == 0 and iteration != max_iter - 1:
                test(config, model, epoch=epoch, iteration=iteration)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()
            
            # Logger
            if is_log_iter:
                for writer in writers:
                    writer.write()

            # Checkpointer step
            periodic_checkpointer.step(iteration, epoch=epoch)
        
        writer.close()
        # Do test at last 
        test(config, model, epoch=epoch, iteration=iteration)
        comm.synchronize()

@torch.no_grad()
def test(config, model, epoch=None, iteration=None):
    results = OrderedDict()
    model_name = osp.basename(config.model.pretrained).split('.')[0]
    for dataset_name in config.datasets.test:
        if epoch is not None and iteration is not None:
            eval_out_dir = osp.join(config.output_dir, f'test_epoch_{epoch:02d}_iter_{iteration:06d}', dataset_name)
        else:
            eval_out_dir = osp.join(config.output_dir, f'test_{model_name}', dataset_name)
        evaluator = build_evaluator(config, dataset_name, eval_out_dir)
        data_loader = build_gdrn_test_loader(config, dataset_name, train_objs=evaluator.train_objs)
        results_i = gdrn_inference_on_dataset(config, model, data_loader, evaluator)
        results[dataset_name] = results_i
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def build_evaluator(cfg, dataset_name, output_folder=None):
    """Create evaluator(s) for a given dataset.

    This uses the special metadata "evaluator_type" associated with each
    builtin dataset. For your own dataset, you can simply create an
    evaluator manually in your script and do not have to worry about the
    hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = osp.join(cfg.output_dir, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

    _distributed = comm.get_world_size() > 1
    dataset_meta = MetadataCatalog.get(cfg.datasets.train[0])
    train_obj_names = dataset_meta.objs
    if evaluator_type == "bop":
        gdrn_eval_cls = GDRN_EvaluatorCustom
        return gdrn_eval_cls(
            cfg, dataset_name, distributed=_distributed, output_dir=output_folder, train_objs=train_obj_names
        )
    else:
        raise NotImplementedError(
            "Evaluator type {} is not supported for dataset {}".format(evaluator_type, dataset_name)
        )

def cleanup():
    dist.destroy_process_group()
 
def setup_ddp(config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

    try:
        world_size = len(config.gpus.split(','))
        mp.spawn(train_worker, nprocs=world_size, args=(world_size, config,))
    except Exception as e:
        print(f'Exception: {e}')
        cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)

    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    setup_ddp(config)
