from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.distill = config['distill']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)          

        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, config['num_labels'])  ## changed
                )

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))               
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      
            self.cls_head_m = nn.Sequential(
                      nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                      nn.ReLU(),
                      nn.Linear(self.text_encoder.config.hidden_size, config['num_labels'])  ## changed
                    )

            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.cls_head,self.cls_head_m],
                               ]
            self.copy_params()        
            self.momentum = 0.995
            
            
    def forward(self, image, text, targets, alpha=0, train=True): 

        captions, questions = text
        assert image is not None or captions is not None, "at least none modality should be presented"
        
        if image is not None:
            image_embeds = self.visual_encoder(image)
        if captions is not None:
            caption_output = self.text_encoder(captions.input_ids,
                                            attention_mask = captions.attention_mask,
                                            return_dict = True, mode = 'text')          
            caption_embeds = caption_output.last_hidden_state

        if image is not None and captions is not None:
            multimodal_embeds = torch.cat((image_embeds, caption_embeds), axis=1)
        elif image is not None:
            multimodal_embeds = image_embeds
        elif captions is not None:
            multimodal_embeds = caption_embeds
        multimodal_atts = torch.ones(multimodal_embeds.size()[:-1],dtype=torch.long).to(multimodal_embeds.device)  
        # print(image_embeds.shape)
        # print(caption_embeds.shape)
        # print(multimodal_embeds.shape, multimodal_atts.shape)
        
        if train:
            output = self.text_encoder(questions.input_ids, 
                                       attention_mask = questions.attention_mask, 
                                       encoder_hidden_states = multimodal_embeds,
                                       encoder_attention_mask = multimodal_atts,        
                                       return_dict = True
                                      )         
            prediction = self.cls_head(output.last_hidden_state[:,0,:])                
            if self.distill:                
                with torch.no_grad():
                    self._momentum_update()
            
                    if image is not None:
                        image_embeds_m = self.visual_encoder_m(image)
                    if captions is not None:
                        caption_output_m = self.text_encoder_m(captions.input_ids,
                                                        attention_mask = captions.attention_mask,
                                                        return_dict = True, mode = 'text')          
                        caption_embeds_m = caption_output_m.last_hidden_state

                    if image is not None and captions is not None:
                        multimodal_embeds_m = torch.cat((image_embeds_m, caption_embeds_m), axis=1)
                    elif image is not None:
                        multimodal_embeds_m = image_embeds_m
                    elif captions is not None:
                        multimodal_embeds_m = caption_embeds_m
                    
                    output_m = self.text_encoder_m(questions.input_ids, 
                                               attention_mask = questions.attention_mask, 
                                               encoder_hidden_states = multimodal_embeds_m,
                                               encoder_attention_mask = multimodal_atts,        
                                               return_dict = True
                                              )           
                    prediction_m = self.cls_head_m(output_m.last_hidden_state[:,0,:])   

                loss = (1-alpha)*F.cross_entropy(prediction, targets) - alpha*torch.sum(
                    F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()
            else:
                loss = F.cross_entropy(prediction, targets)                
            return loss 
            
        else:
            output = self.text_encoder(questions.input_ids, 
                                       attention_mask = questions.attention_mask, 
                                       encoder_hidden_states = multimodal_embeds,
                                       encoder_attention_mask = multimodal_atts,        
                                       return_dict = True
                                      )         
            prediction = self.cls_head(output.last_hidden_state[:,0,:])                        
            return prediction
 


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                

