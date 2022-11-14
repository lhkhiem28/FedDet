
import os, sys
from libs import *

def client_fit_fn(
    fit_loaders, 
    model, 
    num_epochs, 
    optimizer, 
    device = torch.device("cpu"), 
    save_ckp_dir = "./", 
    fitting_verbose = True, 
):
    print("\nStart Fitting ...\n" + " = "*16)
    model = model.to(device)

    for epoch in tqdm.tqdm(range(1, num_epochs + 1), disable = fitting_verbose):
        if fitting_verbose:
            print("epoch {:2}/{:2}".format(epoch, num_epochs) + "\n" + " - "*16)
        model.train()
        running_loss = 0.0
        for images, labels in tqdm.tqdm(fit_loaders["fit"], disable = not fitting_verbose):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = compute_loss(
                logits, labels, 
                model, 
            )[0]

            loss.backward()
            optimizer.step(), optimizer.zero_grad()

            running_loss = running_loss + loss.item()*images.size(0)
        fit_loss = running_loss/len(fit_loaders["fit"].dataset)
        if fitting_verbose:
            print("{:<8} - loss:{:.4f}".format(
                "fit", fit_loss, 
            ))

        with torch.no_grad():
            model.eval()
            running_classes, running_statistics = [], []
            for images, labels in tqdm.tqdm(fit_loaders["evaluate"], disable = not fitting_verbose):
                images, labels = images.to(device), labels.to(device)
                labels[:, 2:] = xywh2xyxy(labels[:, 2:])
                labels[:, 2:] = labels[:, 2:]*int(fit_loaders["evaluate"].dataset.image_size)

                logits = model(images)
                logits = non_max_suppression(
                    logits, 
                    conf_thres = 0.1, iou_thres = 0.5, 
                )

                running_classes, running_statistics = running_classes + labels[:, 1].tolist(), running_statistics + get_batch_statistics(
                    [logit.cpu() for logit in logits], labels.cpu(), 
                    0.5, 
                )
        evaluate_map = ap_per_class(
            *[np.concatenate(stats, 0) for stats in list(zip(*running_statistics))], 
            running_classes, 
        )[2].mean()
        if fitting_verbose:
            print("{:<8} - loss:{:.4f}".format(
                "evaluate", evaluate_map, 
            ))

    torch.save(model, "{}/client.ptl".format(save_ckp_dir))
    print("\nFinish Client Fitting ...\n" + " = "*16)
    return {
        "fit_loss":fit_loss, 
        "evaluate_map":evaluate_map, 
    }