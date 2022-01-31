import torch 
import torch.nn as nn 

class AttentivePooling(nn.Module):
    """
    Attentive Pooling module incoporate attention mask 
    """
    def __init__(self, feature_dim, output_dim, **kwargs):
        super(AttentivePooling, self).__init__()

        self.feature_transform = nn.Linear(feature_dim, output_dim)
        self.W_a = nn.Linear(output_dim, output_dim)
        self.W = nn.Linear(output_dim, 1)
        self.act_fn = nn.ReLU()
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask):
        """
        input:
        batch_rep : size (B, T, H), B: batch size, T: sequence length, H: Hidden dimension
        att_mask:  size (B, T),     Attention Mask logits
        
        attention_weight:
        att_w : size (B, T, 1)
        
        return:
        utter_rep: size (B, H)
        """
        batch_rep  = self.feature_transform(batch_rep)
        att_logits = self.W(self.act_fn(self.W_a(batch_rep))).squeeze(-1)
        att_logits = att_mask + att_logits # masked out frames recieves ~0% prob. 
        # compute attention att_w
        # softmax over segment dimension i.e. take the most representation frame to represent word 
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        # apply att_w to input
        segment_rep = torch.sum(batch_rep * att_w, dim=1) 

        return segment_rep

if __name__ == '__main__':
    x = torch.randn(torch.Size([10, 27, 60, 768]))
    attn_mask = torch.where(x == 0, -100000, 0)
    attn_mask = attn_mask[:, :, :, 0].squeeze(-1)

    # combine sentence-level and segment-level into batch dimensions
    x_resized = x.reshape(-1, 60, 768) # B, T, F
    attn_mask_resized = attn_mask.reshape(-1, 60) # B, T
    m = AttentivePooling(768, 768)
    y = m(x_resized, attn_mask_resized)
    y = y.reshape(10, 27, 768)
    print(y.shape)
