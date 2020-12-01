import s3fs
s3 = s3fs.S3FileSystem(anon=True)

from torchvision import datasets, transforms, models

resnet = models.resnet50(pretrained=True)

with s3.open('s3://saturn-public-data/dogs/imagenet1000_clsidx_to_labels.txt') as f:
    classes = [line.strip() for line in f.readlines()]
    
    
from PIL import Image
to_pil = transforms.ToPILImage()

with s3.open("s3://saturn-public-data/dogs/2-dog.jpg", 'rb') as f:
    img = Image.open(f).convert("RGB")

transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(250), 
    transforms.ToTensor()])

img_t = transform(img)