import torch, numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as Data

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w, h = imgs[0].size
    tensor = np.zeros((len(imgs), 3, w, h), dtype=np.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        tensor[i] = np.rollaxis(nump_array, 2)
    tensor = torch.from_numpy(tensor)
    return tensor.contiguous(), targets

def imagenet_loader(datapath, batch_size):
    train_dataset = datasets.ImageFolder(datapath + '/train/',
        transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]))

    train_sampler = Data.distributed.DistributedSampler(train_dataset)
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=12, shuffle=False, drop_last=True, collate_fn=fast_collate)

    val_dataset = datasets.ImageFolder(datapath + '/val/',
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)]))
    
    val_sampler = Data.distributed.DistributedSampler(val_dataset)
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size * 2, sampler=val_sampler,
        num_workers=12, shuffle=False, drop_last=False, collate_fn=fast_collate)
    return train_loader, train_sampler, val_loader




