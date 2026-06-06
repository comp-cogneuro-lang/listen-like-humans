from sched import scheduler
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
import numpy as np
from data import read_config, get_datasets
import models
from training import Trainer
import argparse
import sys, os, json
import wandb
import time

def setup_ddp(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '12355')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    
def parse_args():

    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_cpu', action='store_true', help='run on CPU instead')
    parser.add_argument('--datapath', type=str, 
                        default=os.getenv('DATAPATH', 'dataset'),
                        help='path to dataset directory')
    parser.add_argument('--num_workers', type=int, 
                        default=int(os.getenv('NUM_WORKERS', 8)), 
                        help='number of workers for data loading')
    parser.add_argument('--config_path', type=str,
                        default=os.getenv('CONFIG_PATH', 'experiments/en_words_ku.cfg'),
                        help='path to config file with hyperparameters, etc.')
    parser.add_argument('--num_epochs', type=int, default=None, help='number of training epochs')
    parser.add_argument('--name', type=str, 
                        default=os.getenv('EXPERIMENT_NAME', 'experiment'), 
                        help='name of the experiment')
    parser.add_argument('--ddp', action='store_true', help='use distributed data parallel')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training (deprecated)')
    args = parser.parse_args()
    return args

def setup_training(config, datapath, num_workers, is_cpu, use_ddp=False, rank=0, use_amp=False):
    train_dataset, valid_dataset, test_dataset = get_datasets(config, datapath,
        config.pretraining_manifest_train,
        config.pretraining_manifest_dev,
        config.pretraining_manifest_test, num_workers, use_ddp=use_ddp)

    # Initialize model
    model = getattr(models, config.type)(config=config)
    if torch.cuda.is_available() and not is_cpu:
        if use_ddp:
            model.cuda(rank)
        else:
            model.cuda()
    
    # Wrap model with DDP
    if use_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank)
        
    if rank == 0:  # Only print on main process
        print(model)
    
    # Train the model
    trainer = Trainer(model=model, config=config, use_ddp=use_ddp, rank=rank, use_amp=use_amp)

    return trainer, train_dataset, valid_dataset, test_dataset

class NoamScheduler:
    def __init__(self, optimizer, learning_rate, warmup_steps=4000):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        step = self.step_num
        lr = self.learning_rate * (self.warmup_steps ** 0.5) * min(step * (self.warmup_steps ** -1.5), step ** -0.5)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
if __name__ == "__main__":

    args = parse_args()
    
    # Setup DDP if requested
    use_ddp = args.ddp
    if use_ddp:
        local_rank_env = os.environ.get('LOCAL_RANK')
        rank = int(os.environ.get('RANK', args.local_rank))
        world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
        if local_rank_env is not None:
            rank = int(local_rank_env)
        setup_ddp(rank, world_size)
    else:
        rank = 0
        world_size = 1

    config_path = args.config_path
    is_cpu = args.is_cpu
    datapath = args.datapath
    num_workers = args.num_workers
    num_epochs = args.num_epochs

    # Read config file
    config = read_config(config_path, args.name)
    config.datapath = datapath
    if num_epochs is not None:
        config.pretraining_num_epochs = num_epochs

    set_seed(config.seed)
    if torch.cuda.is_available() and not is_cpu:
        torch.backends.cudnn.benchmark = True
    use_amp = (not is_cpu) and torch.cuda.is_available() and os.getenv("USE_AMP", "1") == "1"
 
    trainer, train_dataset, valid_dataset, test_dataset = setup_training(
        config, datapath, num_workers, is_cpu, use_ddp=use_ddp, rank=rank, use_amp=use_amp
    )

    # Initialize wandb (only on main process)
    if os.getenv('USE_WANDB') == '1' and rank == 0:
        wandb.init(project="HSR-Benchmark", name=args.name, config=vars(config))
    else:
        wandb = None
 
    if rank == 0:
        print('Learning rate: %.10f' %trainer.optimizer.param_groups[0]['lr'])

    trainer.scheduler = NoamScheduler(
        trainer.optimizer,
        learning_rate=config.pretraining_lr,
        warmup_steps=200
        )
    best_val_accuracy = 0.
    
    # Determine starting epoch
    start_epoch = 0
    
    for epoch in range(start_epoch, config.pretraining_num_epochs):
        if use_ddp:
            train_dataset.loader.sampler.set_epoch(epoch)  # Important for proper shuffling
        start_time = time.time()
        print("========= Epoch %d of %d =========" % (epoch+1, config.pretraining_num_epochs))

        if epoch % config.pretraining_eval_interval == 0 and epoch !=0:
            train_losses = trainer.train(train_dataset, logger=wandb, calculate_accuracy=True)
            valid_losses, valid_accuracy = trainer.test(valid_dataset, logger=wandb)
            if (epoch==0) or (valid_accuracy["words"]>best_val_accuracy):
                print('Saving checkpoint')
                trainer.save_checkpoint(i_checkpoint=epoch)
                best_val_accuracy = valid_accuracy["words"]
        else:
            train_losses = trainer.train(train_dataset, logger=wandb)
            valid_losses, valid_accuracy = { key: -1 for key in trainer.losses}, { key: -1 for key in trainer.losses}

        print("========= Results: epoch %d of %d =========" % (epoch+1, config.pretraining_num_epochs))
        print('Learning rate: %.10f' %trainer.optimizer.param_groups[0]['lr'])
        print(f'Training Loss:{train_losses}, Best val phoneme accuracy:{best_val_accuracy:.4f}')
        for loss_label in  trainer.losses:
            print(" |- %s: train loss: %.4f| valid loss: %.4f| valid accuracy: %.4f" % (loss_label, train_losses[loss_label], valid_losses[loss_label], valid_accuracy[loss_label]) )
        print("Time elapsed: %.2f seconds\n" % ((time.time() - start_time)))
        
    if wandb is not None:
        wandb.finish()
    
    # Cleanup DDP
    if use_ddp:
        cleanup_ddp()
