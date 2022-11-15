
import os, sys
from libs import *

def metrics_aggregation_fn(metrics):
    fit_losses = [metric["fit_loss"] for _, metric in metrics]
    evaluate_maps = [metric["evaluate_map"] for _, metric in metrics]
    aggregated_metrics = {
        "fit_loss":sum(fit_losses)/len(fit_losses), 
        "evaluate_map":sum(evaluate_maps)/len(evaluate_maps), 
    }

    return aggregated_metrics

class FedAvg(fl.server.strategy.FedAvg):
    def __init__(self, 
        initial_model, 
        save_ckp_dir, 
        *args, **kwargs
    ):
        self.initial_model = initial_model
        self.save_ckp_dir = save_ckp_dir
        super().__init__(*args, **kwargs)

        self.best_map = 0

    def aggregate_fit(self, 
        server_round, 
        results, failures, 
    ):
        aggregated_metrics = metrics_aggregation_fn(
            [(result.num_examples, result.metrics) for _, result in results]
        )
        wandb.log({"fit_loss":aggregated_metrics["fit_loss"]}, step = server_round)
        wandb.log({"evaluate_map":aggregated_metrics["evaluate_map"]}, step = server_round)

        aggregated_parameters, results = super().aggregate_fit(
            server_round, 
            results, failures, 
        )
        if aggregated_parameters is not None and self.best_map < aggregated_metrics["evaluate_map"]:
            self.initial_model.load_state_dict(
                collections.OrderedDict(
                    {key:torch.tensor(value) for key, value in zip(self.initial_model.state_dict().keys(), fl.common.parameters_to_ndarrays(aggregated_parameters))}
                ), 
                strict = True, 
            )
            torch.save(self.initial_model, "{}/server.ptl".format(self.save_ckp_dir))

            self.best_map = aggregated_metrics["evaluate_map"]

        return aggregated_parameters, {}