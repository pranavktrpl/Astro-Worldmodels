import torch
from torchvision.transforms import v2

class TransformImage():
    def __init__(self, Vl, Vg):
        # self.image = image
        self.Vl = Vl
        self.Vg = Vg
        self.V = Vl + Vg
        self.aug_config = v2.Compose(
            [
                ## Rethink this cropping, we need global and local crops with specific counts, Vl, and Vg. And Vl+Vg = V.
                v2.RandomResizedCrop(256, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8), #color jittering - brightness, contrast, saturation, hue,, 
                v2.RandomGrayscale(p=0.2), #convert to grayscale with prob 0.2
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),#(default prob - 0.5), #gaussian blur with prob 0.5
                v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomRotation(degrees=10),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True), #convert pixel values from 0-255 to float32 and scale to [0, 1]. NNs work better with small floating points.
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #normalize the image to have mean 0 and std 1. These are ImageNet statistics, will learn a fresh for our train set.
            ]
        )

    def augment_image(self, image):
        img = image.convert("RGB")
        return torch.stack([self.aug_config(img) for _ in range(self.V)]) #This needs to be two diff local and global crops.