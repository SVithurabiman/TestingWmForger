
from CIN.codes.utils.yml import parse_yml,set_random_seed,dict_to_nonedict
import torch
from CIN.codes.models.Network import Network
from torchvision import transforms
import os
from PIL import Image
class CIN_MODEL:
    def __init__(self, yml_path = 'CIN/codes/options/opt.yml',path_checkpoint="CIN/pth/cinNet&nsmNet.pth"):               
        option_yml = parse_yml(yml_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt = dict_to_nonedict(option_yml)
        set_random_seed(10)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
       
        path_in = {'path_checkpoint':path_checkpoint, 
                        #'opt_folder':opt_folder
                        }
        self.network = Network(self.opt,self.device, path_in)
        
        self.noise_choice = self.opt['noise']['option']
               
        self.input_transforms = transforms.Compose([
            transforms.CenterCrop((self.opt['datasets']['H'], self.opt['datasets']['W'])),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        self.cinNet = self.network.cinNet
        #self.network.eval()
        self.cinNet.eval()
        
    @torch.no_grad()
    def _encode_(self,img_list,messages,limit=500):
        imgs=os.listdir(img_list)
        imgs=[os.path.join(img_list,k) for k in imgs]
        org_imgs=[]
        watermarking_imgs=[]
        for i, img in enumerate(imgs[:limit]):
            img_tens = Image.open(img).convert('RGB')
            img_tens = self.input_transforms(img_tens).unsqueeze(0).to(self.device)
            #print(img_tens.dtype, messages[i].dtype)
            message = messages[i].to(self.device).float()
            
            watermarking_img = self.cinNet.module.encoder(img_tens, 2*message-1)        
            org_imgs.append((img_tens.cpu().detach()+1)/2)
            watermarking_imgs.append((watermarking_img.cpu().detach()+1)/2)
        return  watermarking_imgs, org_imgs
    
    @torch.no_grad()
    def _decode_(self,watermarking_imgs):
        decoded_messages=[]
        for img in watermarking_imgs:
            img = img.to(self.device)   
            pre_noise = self.cinNet.module.nsm(img)
            img_fake, msg_fake_1, msg_fake_2, msg_nsm  =self.cinNet.module.test_decoder(img, pre_noise)
            decoded_messages.append(msg_nsm.detach().cpu().numpy().round().clip(0, 1))
        return decoded_messages
        
        
        