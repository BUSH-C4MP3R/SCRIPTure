from PIL import Image
import requests
import torch
from torchvision import transforms
import os
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.blip import blip_decoder

image_size = 384
transform = transforms.Compose([
    transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'
    
model = blip_decoder(pretrained=model_url, image_size=384, vit='large')
model.eval()
model = model.to(device)


from models.blip_vqa import blip_vqa

image_size_vq = 480
transform_vq = transforms.Compose([
    transforms.Resize((image_size_vq,image_size_vq),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 

model_url_vq = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_vqa.pth'
    
model_vq = blip_vqa(pretrained=model_url_vq, image_size=480, vit='base')
model_vq.eval()
model_vq = model_vq.to(device)



def inference(raw_image, model_n, question="", strategy=""):
    if model_n == 'Image Captioning':
        image = transform(raw_image).unsqueeze(0).to(device)   
        with torch.no_grad():
            if strategy == "Beam search":
                caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
            else:
                caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
            return 'caption: '+caption[0]

    else:   
        image_vq = transform_vq(raw_image).unsqueeze(0).to(device)  
        with torch.no_grad():
            answer = model_vq(image_vq, question, train=False, inference='generate') 
        return  'answer: '+answer[0]

#get caption for a single iamge
def get_caption(image_path):
  img = Image.open(image_path)
  return inference(img, "Image Captioning")[9:]

def display(image_path):
  img = mpimg.imread(image_path)
  img = Image.open(image_path)
  plt.imshow(img)
  print("Caption: " + get_caption(image_path))
  
#returns a dictionary with key -> img_path and value -> caption
def get_captions(img_directory, print_status=True):
    #key is img path, value is the caption 
    captions = {}
    length = 0
    for file in os.listdir(img_directory):
      length+=1
    count = 0
    for file in os.listdir(img_directory):
        f = os.path.join(img_directory, file)
        captions[f] = inference(Image.open(f), "Image Captioning")
        if print_status:
          print("Images complete:", str(count) + "/" + str(length))
          print("Caption:", captions[f])
    return captions
#writes dictionary to file, key and value seperated by ':'
def write_to_file(filename, caption_dict):
  with open(filename, "w") as file:
    for i in caption_dict:
      file.write(i + ":" + caption_dict[i])
  file.close()
    
