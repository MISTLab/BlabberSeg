from torchvision import transforms

img_size = 352

def transform(img):
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [.485, .456, .406], std = [.229, .224, .225]), # std. ImageNet stats.
                    transforms.Resize((img_size, img_size), antialias = True),
                ])
              
    return transform
