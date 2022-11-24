
import os, sys
from libs import *
from data import DetImageDataset
from strategies import FedAvg
from engines import server_test_fn

parser = argparse.ArgumentParser()
parser.add_argument("--server_address", type = str, default = "10.42.0.1"), parser.add_argument("--server_port", type = int, default = 8080)
parser.add_argument("--dataset", type = str, default = "VOC2007"), parser.add_argument("--num_clients", type = int, default = 2)
parser.add_argument("--num_rounds", type = int, default = 200)
parser.add_argument("--num_epochs", type = int, default = 5)
args = parser.parse_args()
wandb.login()
wandb.init(
    project = "FedDet" + "-" + args.dataset + "-" + "Testbed", name = "yolov3 - {:2} clients".format(args.num_clients), 
)

initial_model = Darknet("pytorchyolo/configs/yolov3.cfg")
initial_model.load_darknet_weights("../ckps/darknet53.conv.74")
initial_parameters = fl.common.ndarrays_to_parameters(
    [value.cpu().numpy() for key, value in initial_model.state_dict().items()]
)
save_ckp_dir = "../ckps/{}".format(args.dataset)
if not os.path.exists(save_ckp_dir):
    os.makedirs(save_ckp_dir)
fl.server.start_server(
    server_address = "{}:{}".format(args.server_address, args.server_port), 
    config = fl.server.ServerConfig(num_rounds = args.num_rounds), 
    strategy = FedAvg(min_available_clients = args.num_clients, 
        min_fit_clients = args.num_clients, 
        min_evaluate_clients = args.num_clients, 
        initial_model = initial_model, 
        initial_parameters = initial_parameters, 
        save_ckp_dir = save_ckp_dir, 
    )
)
wandb.finish()

dataset = DetImageDataset(
    images_path = "../datasets/VOC2007/test/images", labels_path = "../datasets/VOC2007/test/labels"
    , image_size = 224
    , augment = False
)
test_loader = torch.utils.data.DataLoader(
    dataset, collate_fn = dataset.collate_fn, 
    num_workers = 0, batch_size = 2, 
    shuffle = False, 
)
model = torch.load("../ckps/{}/server.ptl".format(args.dataset))
server_test_fn(
    test_loader, 
    model, 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
)