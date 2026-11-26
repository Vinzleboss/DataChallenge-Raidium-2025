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

def soft_dice_loss(logits, y, A, epsilon = 1.0):
    B = logits.shape[0]
    A_mask = torch.zeros(B, NUM_CLASSES+1, dtype=torch.bool, device = logits.device)
    for i in range(B):
        A_mask[i,A[i]] = True
    A_mask[:,0] = False
    prob_pred = F.softmax(logits, dim = 1)
    H,W = y.shape[1:]
    y_groundtruth = torch.zeros(B,NUM_CLASSES+1, H,W, device=logits.device, dtype=torch.float)
    for c in range(1,NUM_CLASSES+1):
        y_groundtruth[:,c] = (y==c)

    pred = torch.sum(prob_pred, dim=(2,3))
    truth = torch.sum(y_groundtruth, dim=(2,3))
    intersect = torch.sum(prob_pred*y_groundtruth, dim=(2,3))

    dice_c = (2*intersect + epsilon)/(pred+truth+epsilon)

    dice_c_real = A_mask.to(torch.float)*dice_c
    weight = torch.sum(A_mask).clamp(min=1)

    dice_score = 1/weight * torch.sum(dice_c_real)
    return 1 - dice_score
    

def CE_Modified_plus_DiceLoss(logits, y, A) :
    return soft_dice_loss(logits,y,A) + cross_entropy_modified(logits,y,A)




