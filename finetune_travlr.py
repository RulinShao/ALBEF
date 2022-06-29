import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_travlr import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
import wandb
from accelerate import Accelerator
from sklearn.metrics import f1_score

DISTRIBUTED = False

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, accelerator=None):
    data_loader = accelerator.prepare(data_loader)
    # model, optimizer, data_loader = accelerator.prepare(
        # model, optimizer, data_loader)

    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  

    accum_iter = config['accum_iter']
    
    if not config['no_caption'] and not config['no_image']:
        for i,(images, text, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
            images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)
            captions, questions = text
            
            caption_inputs = tokenizer(captions, padding='longest', return_tensors="pt").to(device)   
            question_inputs = tokenizer(questions, padding='longest', return_tensors="pt").to(device)  
            text_inputs = [caption_inputs, question_inputs]
            
            if epoch>0 or not config['warm_up']:
                alpha = config['alpha']
            else:
                alpha = config['alpha']*min(1,i/len(data_loader))

            loss = model(images, text_inputs, targets=targets, train=True, alpha=alpha)
            loss = loss / accum_iter
            
            accelerator.backward(loss)

            if ((i + 1) % accum_iter == 0) or (i + 1 == len(data_loader)):
                optimizer.step()
                optimizer.zero_grad()
                
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(loss=loss.item())
            
            if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
                scheduler.step(i//step_size)   
    
    elif config['no_caption']:
        for i,(images, questions, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
            images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)
            
            question_inputs = tokenizer(questions, padding='longest', return_tensors="pt").to(device)  
            text_inputs = [None, question_inputs]
            
            if epoch>0 or not config['warm_up']:
                alpha = config['alpha']
            else:
                alpha = config['alpha']*min(1,i/len(data_loader))

            loss = model(images, text_inputs, targets=targets, train=True, alpha=alpha)
            loss = loss / accum_iter
            
            accelerator.backward(loss)

            if ((i + 1) % accum_iter == 0) or (i + 1 == len(data_loader)):
                optimizer.step()
                optimizer.zero_grad()
                
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(loss=loss.item())
            
            if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
                scheduler.step(i//step_size)
    
    elif config['no_image']:
        for i,(text, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
            targets = targets.to(device,non_blocking=True)
            captions, questions = text
            
            caption_inputs = tokenizer(captions, padding='longest', return_tensors="pt").to(device)   
            question_inputs = tokenizer(questions, padding='longest', return_tensors="pt").to(device)  
            text_inputs = [caption_inputs, question_inputs]
            
            if epoch>0 or not config['warm_up']:
                alpha = config['alpha']
            else:
                alpha = config['alpha']*min(1,i/len(data_loader))

            loss = model(None, text_inputs, targets=targets, train=True, alpha=alpha)
            loss = loss / accum_iter
            
            accelerator.backward(loss)

            if ((i + 1) % accum_iter == 0) or (i + 1 == len(data_loader)):
                optimizer.step()
                optimizer.zero_grad()
                
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(loss=loss.item())
            
            if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
                scheduler.step(i//step_size)
    else:
        raise AttributeError
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    if not config['no_caption'] and not config['no_image']:
        for images, text, targets in metric_logger.log_every(data_loader, print_freq, header):
            
            if images is not None:
                images = images.to(device,non_blocking=True)
            targets = targets.to(device,non_blocking=True)  
            captions, questions = text
            
            if caption is not None:
                caption_inputs = tokenizer(captions, padding='longest', return_tensors="pt").to(device)
            else:
                caption_inputs = None   
            question_inputs = tokenizer(questions, padding='longest', return_tensors="pt").to(device)  
            text_inputs = [caption_inputs, question_inputs] 

            prediction = model(images, text_inputs, targets=targets, train=False)  
    
            _, pred_class = prediction.max(1)
            accuracy = (targets==pred_class).sum() / targets.size(0)

            metric_logger.meters['acc'].update(accuracy.item(), n=images.size(0))
            
            f1_macro = f1_score(targets.cpu(), pred_class.cpu(), average='macro')
            metric_logger.meters['f1_macro'].update(f1_macro, n=images.size(0))
    
    if config['no_caption']:
        for images, questions, targets in metric_logger.log_every(data_loader, print_freq, header):
            
            if images is not None:
                images = images.to(device,non_blocking=True)
            targets = targets.to(device,non_blocking=True)  
            
            question_inputs = tokenizer(questions, padding='longest', return_tensors="pt").to(device)  
            text_inputs = [None, question_inputs]

            prediction = model(images, text_inputs, targets=targets, train=False)  
    
            _, pred_class = prediction.max(1)
            accuracy = (targets==pred_class).sum() / targets.size(0)

            metric_logger.meters['acc'].update(accuracy.item(), n=images.size(0))
            
            f1_macro = f1_score(targets.cpu(), pred_class.cpu(), average='macro')
            metric_logger.meters['f1_macro'].update(f1_macro, n=images.size(0))
    
    if config['no_image']:
        for text, targets in metric_logger.log_every(data_loader, print_freq, header):
            
            targets = targets.to(device,non_blocking=True)  
            captions, questions = text
            
            caption_inputs = tokenizer(captions, padding='longest', return_tensors="pt").to(device)
            question_inputs = tokenizer(questions, padding='longest', return_tensors="pt").to(device)  
            text_inputs = [caption_inputs, question_inputs] 

            prediction = model(None, text_inputs, targets=targets, train=False)  
    
            _, pred_class = prediction.max(1)
            accuracy = (targets==pred_class).sum() / targets.size(0)

            metric_logger.meters['acc'].update(accuracy.item(), n=captions.size(0))
            
            f1_macro = f1_score(targets.cpu(), pred_class.cpu(), average='macro')
            metric_logger.meters['f1_macro'].update(f1_macro, n=captions.size(0))
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    
    
def main(args, config):
    if args.wandb_log:
        WANDB_PROJECT = config['wandb_project']
        WANDB_RUN_NAME = config['wandb_run_name']
        WANDB_NOTES = config['wandb_notes']
        
        wandb.init(project=WANDB_PROJECT,
                name=WANDB_RUN_NAME,
                notes=WANDB_NOTES)
        wandb_config = wandb.config
        wandb_config.config = config

    utils.init_distributed_mode(args)    
    
    # device = torch.device(args.device)
    accelerator = Accelerator()
    device = accelerator.device
    print(device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    print("random seed: " + str(seed))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    datasets = create_dataset('travlr', config) 
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],is_trains=[True,False,False], 
                                                          collate_fns=[None,None,None])

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)

    if args.wandb_log:
        wandb.watch(model)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        
        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

            for key in list(state_dict.keys()):                
                if 'bert' in key:
                    new_key = key.replace('bert.','')
                    state_dict[new_key] = state_dict[key] 
                    del state_dict[key]
                
        msg = model.load_state_dict(state_dict,strict=False)
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)

    # model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
        
    PROBING = False
    if PROBING:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.cls_head_m.parameters():
            param.requires_grad = True
            
        for param in model.cls_head.parameters():
            param.requires_grad = True

        for name, param in model.named_parameters():
            print(name, param.requires_grad)

        # print(len(list(model.parameters())))
        # print(list(filter(lambda p: p.requires_grad, model.parameters())))
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0
    
    print("Start training")
    start_time = time.time()

    for epoch in range(0, max_epoch):
        model, optimizer = accelerator.prepare(model, optimizer)
        
        if not args.evaluate:
            print("Start training")
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, accelerator=accelerator)  
            
        val_loader = accelerator.prepare(val_loader)
        test_loader = accelerator.prepare(test_loader)
        
        if epoch % 10 == 0 or epoch == (max_epoch - 1):
            val_stats = evaluate(model, val_loader, tokenizer, device, config)
            test_stats = evaluate(model, test_loader, tokenizer, device, config)

        if utils.is_main_process():  
            if args.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }

                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                
            else:    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                            }

                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                    
                save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint.pth'))

                if float(val_stats['acc'])>best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                    best = float(val_stats['acc'])

                if args.wandb_log:
                    log_stats = {k:float(v) for k,v in log_stats.items()}
                    wandb.log(log_stats)
        
        if args.evaluate:
            break
        lr_scheduler.step(epoch+warmup_steps+1)  
        if DISTRIBUTED:
            dist.barrier()   
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)         
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/custom.yaml')
    parser.add_argument('--output_dir', default='output/TraVLR/')  
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', default=False, action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, action='store_true')
    parser.add_argument('--wandb_log', default=False, action='store_true')
    args = parser.parse_args()
    print(args.wandb_log)

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
