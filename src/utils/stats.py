import pandas as pd

def collectStatistics(loss_per_epoch, acc_per_epoch, test_loss_per_epoch, filename):
    df = pd.DataFrame({'epoch_loss': loss_per_epoch,
                       'epoch_acc': acc_per_epoch,
                       'epoch_test_loss': test_loss_per_epoch})
    df.to_csv(filename)
