from PIL import Image
import requests
import torch
from torchvision import transforms
import os
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cohere
import gradio as gr
import string

def cap(t):
    indices = []
    tem = ""
    for j in range(len(t)):
        if t[j] == "." or t[j] == "!" or t[j] == "?":
            if j+2 < len(t):
                indices.append(j+2)
    for j in range(len(t)):
        if j in indices:
            tem += t[j].upper()
        else:
            tem += t[j]
    return tem
def processing(s):
    #create a string[] that holds every sentence
    arr = []
    temp = ""
    fin = ""
    for i in range(len(s)):
        temp += s[i]
        if s[i] == "\n":
            arr.append(temp)
            temp = ""
        if i == len(s)-1:
            arr.append(temp)
    for i in arr:
        t = i
        t = t.strip()
        temp = ""
        #make the first element of the string be the first alpha character
        ind = 0
        for j in range(len(t)):
            if t[j].isalpha():
                ind = j
                break
        t = t[ind:]
        t = t.capitalize()
        # capitalize all words after punctuation 
        t = cap(t)
        #remove some punctuation
        t = t.replace("(", "")
        t = t.replace(")", "")
        t = t.replace("&", "")
        t = t.replace("#", "")
        t = t.replace("_", "")
        
        #remove punctuation if it is not following an alpha character
        temp = ""
        for j in range(len(t)):
            if t[j] in string.punctuation:
                if t[j-1] not in string.punctuation:
                    temp += t[j]
            else:
                temp += t[j]
        fin += temp + "\n"
        #find the last punctuation in fin and return everything before that
    ind = 0
    for i in range(len(fin)):
        if fin[i] == "." or fin[i] == "?" or fin[i] == "!":
            ind = i
    if(ind != 0 and ind != len(fin) - 1):
        return fin[:ind+1]
    else:
        return fin

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

  # Text to Image API

import requests
import base64


#add max tokens a slider

def make_image_and_story(prompt):
  if(prompt is None or prompt == ""):
    host = 'https://dev.paint.cohere.ai/txt2img'
    response = requests.post(host, json={'prompt': 'Random monster', 'n_samples' : 1, 'n_iter' : 1})

    # decode image
    imageBytes = base64.b64decode(response.json()['image']) #decode

    # save to disk
    f = open("sample.png", "wb")
    f.write(imageBytes)
    f.close()

    caption = get_caption("sample.png")

    co = cohere.Client('SD5vY3pwFrA0bBNTnIpp4N02sWhK4vd7mkkcrpXS')
    response = co.generate(prompt=caption, model ='aeb523c3-a79c-48ba-9274-a12ac07492a2-ft', max_tokens=80)

    return Image.open("sample.png"), processing(response.generations[0].text)
  else:
    host = 'https://dev.paint.cohere.ai/txt2img'
    response = requests.post(host, json={'prompt': prompt+", epic", 'n_samples' : 1, 'n_iter' : 1})

    # decode image
    imageBytes = base64.b64decode(response.json()['image']) #decode

    # save to disk
    f = open("sample.png", "wb")
    f.write(imageBytes)
    f.close()

    caption = get_caption("sample.png")
    caption += " " + prompt

    co = cohere.Client('SD5vY3pwFrA0bBNTnIpp4N02sWhK4vd7mkkcrpXS')
    response = co.generate(prompt=caption, model ='aeb523c3-a79c-48ba-9274-a12ac07492a2-ft', max_tokens=80)

    return Image.open("sample.png"), processing(response.generations[0].text)


gr.Interface(fn=make_image_and_story, inputs="text", outputs=["image","text"],title='Fantasy Creature Generator').launch();
