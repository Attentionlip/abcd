import random
import torch
import torchvision
from transformers import BertTokenizer
import torch.nn.functional as F
from torch import nn


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    #tokenizer.silence_token_id = tokenizer.additional_special_tokens_ids[1] 
    return tokenizer

class AudioVisualModel(torch.nn.Module):
    def name(self):
        return "AudioVisualModel"


    def __init__(self, nets):
        super(AudioVisualModel, self).__init__()

        # initialize model
        self.net_lipreading, self.net_facial, self.net_diffwave, self.net_text,self.net_attention,self.net_fusion = nets
        self.tokenizer = init_tokenizer()
        self.text_proj = nn.Linear(768, 512) #original
        
        # classifier guidance null conditioners
        torch.manual_seed(0)        # so we have the same null tokens on all nodes
        self.register_buffer("mouthroi_null", torch.randn(1, 1, 1, 88, 88))  # lips regions frames are 88x88 each
        self.register_buffer("face_null", torch.randn(1, 3, 224, 224))  # face image size is 224x224

    def forward(self, melspec, mouthroi, face_image, text, diffusion_steps, cond_drop_prob):
        # classifier guidance
        batch = melspec.shape[0]
        if cond_drop_prob > 0:
            prob_keep_mask = self.prob_mask_like((batch, 1, 1, 1, 1), 1.0 - cond_drop_prob, melspec.device)
            _mouthroi = torch.where(prob_keep_mask, mouthroi, self.mouthroi_null)
            _face_image = torch.where(prob_keep_mask.squeeze(1), face_image, self.face_null)
        else:
            _mouthroi = mouthroi
            _face_image = face_image

        # pass through visual stream and extract lipreading features
        lipreading_feature = self.net_lipreading(_mouthroi)
        
        # pass through visual stream and extract identity features
        identity_feature   = self.net_facial(_face_image)

        max_length = lipreading_feature.shape[-1]
        text = self.tokenizer(text, padding='max_length', truncation=True, max_length=max_length, 
                              return_tensors="pt").to(melspec.device)  
        text_output = self.net_text(text.input_ids, attention_mask = text.attention_mask,  return_dict = True)
        text_feat = self.text_proj(text_output.last_hidden_state).permute(0,2,1)   # [128,256]


        video_att, text_att, vt_att , tv_att= self.net_attention(lipreading_feature, text_feat, text_mask = None)

        
        video_att = video_att.permute(0,2,1).unsqueeze(2)
        text_att  =  text_att.permute(0,2,1).unsqueeze(2)
        vt_att    =    vt_att.permute(0,2,1).unsqueeze(2)
        tv_att    =    tv_att.permute(0,2,1).unsqueeze(2)
        # what type of visual feature to use
        identity_feature = identity_feature.repeat(1, 1, 1, lipreading_feature.shape[-1])
        visual_feature   = torch.cat((identity_feature,video_att,text_att, vt_att ,tv_att), dim=1)
        visual_feature   = visual_feature.squeeze(2)  # so dimensions are B, C, num_frames
        

        
        output = self.net_diffwave((melspec, diffusion_steps), cond=visual_feature)
        return output


    @staticmethod
    def prob_mask_like(shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device=device, dtype=torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device=device, dtype=torch.bool)
        else:
            return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
