def save_better_model(current,last,save_path,current_model):
    import torch
    if current > last:
        last = current
        torch.save(
                current_model.state_dict(),
                save_path)
        print("Best model save with: {} in save path:{}".format(last,save_path))
    return last

def save_checkpoint():
    torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                                'optimizer': optimizer.state_dict(),'alpha': loss.alpha, 'gamma': loss.gamma},
                               checkpoint_path + '/m-' + launchTimestamp + '-' + str("%.4f" % lossMIN) + '.pth.tar')

