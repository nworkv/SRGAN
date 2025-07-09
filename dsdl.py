
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tt
from PIL import Image

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def denorm(img_tensors):
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    return img_tensors * stats[1][0] + stats[0][0]

def transform(image: Image, image_size: int, HL = 1 ):
    """
    Parameters
    ----------
    image : Image
        Any size image.
    image_size : int
        size to crop HR image
        size / 4 to crop LR image
    HL : TYPE, optional
        flag to control transformations. The default is 1.

    Returns
    -------
    Image.

    """
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform_HR=tt.Compose([tt.Resize(image_size),
                             tt.CenterCrop(image_size),
                             tt.ToTensor(),
                             tt.Normalize(*stats)])

    transform_LR=tt.Compose([tt.Resize(image_size // 4),
                             tt.CenterCrop(image_size // 4),
                             tt.ToTensor(),
                             tt.Normalize(*stats)])
    if HL == 1:
        image = transform_HR(image)
    else:
        image = transform_LR(image)
    return image
    

#Датасет изображений
class ImageDataset(Dataset):
    def __init__(self, image_filenames, image_size: int):
        self.image_filenames = image_filenames
        self.image_size = image_size

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index])
        Image_HR = transform(image, self.image_size, 1)
        Image_LR = transform(image, self.image_size, 0)

        return (Image_HR, Image_LR)

    def __len__(self):
        return len(self.image_filenames)


class ImageDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dataset, batch_size, device):
        self.dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)