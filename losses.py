import torch
import torch.nn.functional as F
from config import NUM_CLASSES

def cross_entropy_modified(logits, y, A):
    B = logits.shape[0]
    A_mask = torch.zeros(B,NUM_CLASSES+1, dtype=torch.bool, device = logits.device)
    for i in range(B):
        A_mask[i, A[i]] = True

    A_maskextend = A_mask[:,:,None,None]
    logits_Abarre = logits.masked_fill(A_maskextend,float('-inf'))
    lse_Abarre = torch.logsumexp(logits_Abarre, dim=1)
    lse_all = torch.logsumexp(logits, dim=1)
    neglogSall = lse_all - lse_Abarre

    y0 = y == 0
    neglogS = torch.sum(neglogSall*y0, dim=(1,2))
    weights = torch.sum(y0, dim=(1,2))
    modifiedentropy = 1/weights.clamp(min=1) * neglogS

    ydiff0 = y !=0
    classicentropall = F.cross_entropy(logits,y, reduction="none")

    classicentrop = torch.sum(classicentropall*ydiff0, dim=(1,2))
    weightsent = torch.sum(ydiff0, dim=(1,2))
    entrop = 1/weightsent.clamp(min=1) * classicentrop

    lossall = entrop + modifiedentropy
    loss = torch.sum(lossall) * 1/B

    return loss
