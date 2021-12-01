def early_stop(loss_hist, patience, skip= 0):
    #Skip the first n epochs
    if len(loss_hist) <= skip:
        return False
    loss_hist = loss_hist[skip:]
    count = 0
    min_loss = float('inf')
    for i in range(1,len(loss_hist)):
        if loss_hist[i] >= min_loss:
            count += 1
        elif loss_hist[i] < min_loss:
            min_loss = loss_hist[i]
            count = 0
    if count >= patience:
        print('Stop the training')
        return True
    else:
        return False