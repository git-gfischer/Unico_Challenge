class BaseTrainer():
    def __init__(self, cfg, network, optimizer, loss_fn, device, trainloader, valloader, writer, lr_scheduler, report): 
        self.cfg = cfg
        self.network = network
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.trainloader = trainloader
        self.valloader = valloader
        self.writer = writer
        self.loss_fn = loss_fn
        self.report = report