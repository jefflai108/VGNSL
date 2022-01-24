import torch 
import s3prl.hub as hub

model_0 = getattr(hub, 'fbank')()  # use classic FBANK
model_1 = getattr(hub, 'modified_cpc')()  # build the CPC model with pre-trained weights
model_2 = getattr(hub, 'tera')()  # build the TERA model with pre-trained weights
model_3 = getattr(hub, 'wav2vec2')()  # build the Wav2Vec 2.0 model with pre-trained weights
model_3 = getattr(hub, 'hubert')()  # build the Wav2Vec 2.0 model with pre-trained weights

device = 'cuda'  # or cpu
model_3 = model_3.to(device)
wavs = [torch.randn(160000, dtype=torch.float).to(device) for _ in range(16)]
with torch.no_grad():
    reps = model_3(wavs)["hidden_states"] # (batch_size, max_sequence_length_of_batch, hidden_size)
    for i in range(len(reps)): 
        print(reps[i].shape) 
