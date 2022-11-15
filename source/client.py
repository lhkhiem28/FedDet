
import os, sys
from libs import *
from data import DetImageDataset
from engines import client_fit_fn

class Client(fl.client.NumPyClient):
    def __init__(self, 
        fit_loaders, 
        model, 
        num_epochs, 
        num_rounds, 
        optimizer, 
        lr_scheduler, 
        device = torch.device("cpu"), 
        save_ckp_dir = "./", 
        fitting_verbose = True, 
    ):
        self.fit_loaders = fit_loaders
        self.model = model
        self.num_epochs = num_epochs
        self.num_rounds = num_rounds
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.save_ckp_dir = save_ckp_dir
        self.fitting_verbose = fitting_verbose

        self.model = self.model.to(device)
        self.round = 1

    def get_parameters(self, 
        config, 
    ):
        self.model.train()
        parameters = [value.cpu().numpy() for key, value in self.model.state_dict().items()]
        return parameters

    def set_parameters(self, 
        parameters, 
    ):
        self.model.train()
        self.model.load_state_dict(
            collections.OrderedDict(
                {key:torch.tensor(value) for key, value in zip(self.model.state_dict().keys(), parameters)}
            ), 
            strict = True, 
        )
    def fit(self, 
        parameters, config, 
    ):
        self.set_parameters(parameters)
        self.model.train()
        if self.round <= int(0.08*self.num_rounds):
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.model.hyperparams["lr"]*self.round/(int(0.08*self.num_rounds))
        else:
            self.lr_scheduler.step()
        results = client_fit_fn(
            self.fit_loaders, 
            self.model, 
            self.num_epochs, 
            self.optimizer, 
            self.device, 
            self.save_ckp_dir, 
            self.fitting_verbose, 
        )
        self.round += 1
        return self.get_parameters(config = {}), len(fit_loaders["fit"].dataset), results
    def evaluate(self, 
        parameters, config, 
    ):
        return float(len(fit_loaders["evaluate"].dataset)), len(fit_loaders["evaluate"].dataset), {}

parser = argparse.ArgumentParser()
parser.add_argument("--server_address", type = str, default = "127.0.0.1"), parser.add_argument("--server_port", type = int)
parser.add_argument("--dataset", type = str, default = "VOC2007"), parser.add_argument("--cid", type = int)
parser.add_argument("--num_rounds", type = int, default = 250)
parser.add_argument("--num_epochs", type = int, default = 5)
args = parser.parse_args()

datasets = {
    "fit":DetImageDataset(
        images_path = "../datasets/VOC2007/clients/client_{}/fit/images".format(args.cid), labels_path = "../datasets/VOC2007/clients/client_{}/fit/labels".format(args.cid)
        , image_size = 416
        , augment = True
        , multiscale = True
    ), 
    "evaluate":DetImageDataset(
        images_path = "../datasets/VOC2007/clients/client_{}/evaluate/images".format(args.cid), labels_path = "../datasets/VOC2007/clients/client_{}/evaluate/labels".format(args.cid)
        , image_size = 416
        , augment = False
        , multiscale = False
    ), 
}
fit_loaders = {
    "fit":torch.utils.data.DataLoader(
        datasets["fit"], collate_fn = datasets["fit"].collate_fn, 
        num_workers = 1, batch_size = 8, 
        shuffle = True, 
    ), 
    "evaluate":torch.utils.data.DataLoader(
        datasets["evaluate"], collate_fn = datasets["evaluate"].collate_fn, 
        num_workers = 1, batch_size = 8, 
        shuffle = False, 
    ), 
}
model = Darknet("pytorchyolo/configs/yolov3.cfg")
model.load_darknet_weights("../ckps/darknet53.conv.74")
optimizer = optim.Adam(
    model.parameters(), 
    lr = model.hyperparams["lr"], weight_decay = model.hyperparams["weight_decay"], 
)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    eta_min = 0.01*model.hyperparams["lr"], T_max = int(0.92*args.num_rounds), 
)
save_ckp_dir = "../ckps/{}".format(args.dataset)
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)
client = Client(
    fit_loaders, 
    model, 
    args.num_epochs, 
    args.num_rounds, 
    optimizer, 
    lr_scheduler, 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    save_ckp_dir = save_ckp_dir, 
    fitting_verbose = True, 
)
fl.client.start_numpy_client(
    server_address = "{}:{}".format(args.server_address, args.server_port), 
    client = client, 
)