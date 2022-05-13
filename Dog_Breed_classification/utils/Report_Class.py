import os 
import time

class Report_class():
    def __init__(self,cfg,path):
        self.cfg = cfg
        self.timestamp = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
        tmp = self.cfg['model']['base'] + "_" + self.timestamp + ".txt"
        self.filename = os.path.join(path,tmp)
        self.report_header()
    #===================================================    
    def report_header(self):
        file = open(self.filename, 'a')

        # Timestamp // Path Dataset // Model // Batch Size // Optimizer // Loss Funciton // Scheduler // Learning Rate
        file.write('Timestamp: ' + self.timestamp + '\n')
        file.write('Path Dataset: ' + self.cfg['dataset']['root'] + '\n')
        file.write('Model: ' + self.cfg['model']['base'] + '\n')
        file.write('Batch Size: ' + str(self.cfg['test']['batch_size']) + '\n')
        file.write('Optimizer: ' + self.cfg['train']['optimizer'] + '\n')
        file.write('Loss Function: ' + self.cfg['train']['loss_fn'] + '\n')
        if(self.cfg['scheduler']['type'] is None):  file.write('Scheduler[type/step/decay]: None' + '\n')
        else : file.write('Scheduler[type/step/decay]: ' + self.cfg['scheduler']['type'] + '/' + str(self.cfg['scheduler']['step']) + '/' + str(self.cfg['scheduler']['decay']) + '\n')
        file.write('Learning Rate: ' + str(self.cfg['train']['lr']) + '\n')
        file.write('Epochs: ' + str(self.cfg['train']['num_epochs']) + '\n')
        file.write('Epoch,Train_Loss,Val_Loss,Train_acc,Val_acc,lr' + '\n')
        file.close()
    #===================================================
    def report_metrics(self,epoch, train_loss, val_loss, train_acc, val_acc, lr):
        file = open(self.filename, 'a')
        file.write(str(epoch) + ',' + str(round(train_loss,4)) + ',' + str(round(val_loss,4)) + ',' + str(round(train_acc,4)) + ',' + str(round(val_acc,4)) + ',' + str(lr) + '\n')       
        file.close()
