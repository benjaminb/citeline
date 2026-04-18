import torch
from citeline.nn.models import BaselineMLPEmbeddingMapper

model = BaselineMLPEmbeddingMapper()                                                   
model.load_state_dict(torch.load("data/nn_models/basic_sched/addprev02_orig_best.pt/best_model.pth"))
model.eval()                                                                           
torch.jit.script(model).save("data/nn_models/basic_sched/addprev02_orig_best.pt/best_model_scripted.pt")  
