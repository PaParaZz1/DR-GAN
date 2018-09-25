import collections
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

def my_collate(batch):
	if isinstance(batch[0], collections.Sequence):
		return [default_collate(b) for b in batch]
	return default_collate(batch)
def CreateDataLoader(opt):
    """
    Return the dataloader according to the opt.
    """
    import sys
    sys.path.append('/mnt/lustre/niuyazhe/nyz/DG-GAN/data')
    from dataset import FDDataset
    from dataset import MyDataset
    transform = transforms.Compose([
        transforms.Resize((100, 100)),     
        transforms.RandomCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    single = True if opt.model=='single' else False
    if opt.dataset == "cfp": 
        dataset = FDDataset(root=opt.dataroot, train=opt.is_Train, transform=transform, single=single)
    else: 
        dataset = MyDataset(root=opt.testroot, train=opt.is_Train, transform=transform, single=single)


    dataloader = DataLoader(dataset, batch_size=opt.batchsize, shuffle=opt.is_Train, num_workers=4, collate_fn=my_collate)
    return dataloader
