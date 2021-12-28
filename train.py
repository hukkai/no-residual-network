import torch, time, argparse, os
import models, tools
import torchvision
torch.distributed.init_process_group(backend="nccl")
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='SeqNet50')
parser.add_argument('--datapath', type=str, default='./data/')
parser.add_argument('--base_lr', type=float, default=0.02)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--warmup_epoch', type=int, default=5)

parser.add_argument('--zero_init', type=int, default=1)

parser.add_argument('--trial_id', type=str, default='trial1')
parser.add_argument('--saved_model', type=str, default='ckpt.t7')
parser.add_argument('--local_rank', type=int)

args = parser.parse_args()

local_rank = args.local_rank
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
args.num_gpus = torch.distributed.get_world_size()

import builtins as __builtin__
builtin_print = __builtin__.print

def print(*args, **kwargs):
    if local_rank == 0:
        builtin_print(*args, **kwargs)

__builtin__.print = print


################################################
####################  Resume  ##################
################################################
print(args)
resume = False
start_epoch = 0
last_logs = [] # TODO
checkpoint_path = args.trial_id + '/ckpt.t7'

if local_rank == 0 and not os.path.exists(args.trial_id):
    os.mkdir(args.trial_id)

if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, 'cpu')
    resume = True
    backbone_ckpt = ckpt['backbone']
    optimizer_ckpt = ckpt['optimizer']
    start_epoch = ckpt['start_epoch']
    current_iter = ckpt['current_iter']
    last_logs = ckpt['epoch_logs']
    scaler_ckpt = ckpt['scaler']


if 'resnet' in args.model:
    model = getattr(torchvision.models, args.model)(zero_init_residual=args.zero_init)
else:
    model = getattr(models, args.model)()
if resume:
    model.load_state_dict(backbone_ckpt)

model = model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model,
      device_ids=[local_rank], output_device=local_rank)


num_gpus = torch.distributed.get_world_size()
assert args.batch_size % num_gpus == 0
mini_batch = args.batch_size // num_gpus
train_loader, train_sampler, test_loader = tools.imagenet_loader(datapath=args.datapath, batch_size=mini_batch)

criterion = torch.nn.CrossEntropyLoss()


num_layers = 23 if args.model == 'SeqNet101' else 6

if 'resnet' not in args.model:
    cls_para = [i[1] for i in model.named_parameters() if 'classifier' in i[0]]
    layer3_para = [i[1] for i in model.named_parameters() if 'layer3' in i[0]]
    conv_para = [i[1] for i in model.named_parameters() if 'classifier' not in i[0] and 'layer3' not in i[0]]
    optimizer = torch.optim.SGD(
                [{'params': conv_para, 'lr': args.base_lr * args.batch_size / 256},
                 {'params': layer3_para, 'lr': args.base_lr * args.batch_size / 256 * 6 / num_layers},
                 {'params': cls_para, 'lr': 0.1 * args.batch_size / 256}],
                momentum=0.9, weight_decay=args.weight_decay)
elif 'resnet' in args.model:
    optimizer = torch.optim.SGD(model.parameters(),
        lr=0.1 * args.batch_size / 256,
        momentum=0.9, weight_decay=args.weight_decay)



scheduler = tools.warmup_scheduler(iter_per_epoch=len(train_loader),
     max_epoch=args.num_epochs, warmup_epoch=args.warmup_epoch)
scaler = torch.cuda.amp.GradScaler()

if resume:
    optimizer.load_state_dict(optimizer_ckpt)
    scheduler.current_iter = current_iter
    scheduler.base_lr = optimizer_ckpt['param_groups'][0]['initial_lr']
    scaler.load_state_dict(scaler_ckpt)

mean = torch.tensor([123.7, 116.3, 103.5]).view(1,3,1,1).cuda(device, non_blocking=True)
sttdev = torch.tensor([58.4, 57.1, 57.4]).view(1,3,1,1).cuda(device, non_blocking=True)
num_gpus = torch.distributed.get_world_size()
orth_strength = num_gpus * args.weight_decay

print('Begin Training')
for log in last_logs:
    print(log)

t = time.time()
for epoch in range(start_epoch, args.num_epochs):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = correct = total = 0.
    for index, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)
        inputs = inputs.float().sub_(mean).div_(sttdev)
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        orth_loss, count = 0.0, 0
        if 'resnet' not in args.model:
            for p in model.parameters():
                if len(p.shape) == 4 and p.shape[-1] == 1:
                    if count % num_gpus == local_rank:
                        h, w, _, _ = p.shape
                        weight = p.view(h, w)
                        if h <= w:
                            cov = weight @ weight.T
                        else:
                            cov = weight.T @ weight
                        cov = cov.div(cov.diag().mean().add(1e-8).item())
                        reg = cov - torch.eye(min(h, w), device=cov.device)
                        reg = reg.pow(2).sum()
                        orth_loss = orth_loss + reg
                    count += 1

        lr = scheduler.step(optimizer)

        scaler.scale(loss + orth_loss * orth_strength).backward()
        scaler.step(optimizer)
        scaler.update()


        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    num_batches = index + 1.
    val_correct = val_total = 0.
    model.eval()
    for index, (inputs, targets) in enumerate(test_loader):
        with torch.no_grad():
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)
            inputs = inputs.float().sub_(mean).div_(sttdev)
            outputs = model(inputs)

        _, predicted = outputs.max(1)
        val_total += targets.size(0)
        val_correct += predicted.eq(targets).sum().item()


    reduce_tensor = torch.zeros((num_gpus, 6))
    reduce_tensor[local_rank, 0] = correct
    reduce_tensor[local_rank, 1] = total
    reduce_tensor[local_rank, 2] = val_correct
    reduce_tensor[local_rank, 3] = val_total
    reduce_tensor[local_rank, 4] = num_batches
    reduce_tensor[local_rank, 5] = train_loss
    reduce_tensor = reduce_tensor.to(device)
    torch.distributed.all_reduce(reduce_tensor)

    acc_train = 100. * reduce_tensor[:,0].sum().item() / reduce_tensor[:,1].sum().item()
    acc_val = 100. * reduce_tensor[:,2].sum().item() / reduce_tensor[:,3].sum().item()
    train_loss = reduce_tensor[:,5].sum().item() / reduce_tensor[:,4].sum().item()
    used = time.time() - t; t = time.time()
    string = 'Epoch %d, train loss %.2f. Train acc %.2f%%, val acc %.2f%%. LR %.2f, Time: %.2fmins.'%(
                                                    epoch, train_loss, acc_train, acc_val, lr, used/60)
    print(string + '..')
    last_logs.append(string)

    if local_rank == 0:
        state = {
            'backbone': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'start_epoch': epoch + 1,
            'current_iter': scheduler.current_iter,
            'epoch_logs': last_logs,
            'scaler': scaler.state_dict()}

        torch.save(state, checkpoint_path+'_%d.t7'%epoch)
