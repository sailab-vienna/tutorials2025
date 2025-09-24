"""
This file stores all utility functions for x-aware Backdoors!

It inlcudes:
- BadNets: Backdoor attack to add the trigger
- Gradient: Explanation Function

- Calculation of explanation loss
- Filtering Testdata for ASR evaluation
- Evaluation of explanation similarity
- Storing of explanation plots
"""

import os

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torch


def util_tensor_to_numpy_img(t):
    """Convert normalized tensor [3,H,W] -> uint8 numpy [H,W,3]."""
    img = t.permute(1, 2, 0).cpu().numpy()  # HWC, float in [0,1]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def util_denormalize(t, mean, std):
    """Denormalize a tensor image to [0,1] range."""
    mean = torch.tensor(mean, device=t.device).view(-1, 1, 1)
    std = torch.tensor(std, device=t.device).view(-1, 1, 1)
    return t * std + mean

def util_normalize(t, mean, std):
    """Normalize a tensor image from [0,1] range to given mean/std."""
    mean = torch.tensor(mean, device=t.device).view(-1, 1, 1)
    std = torch.tensor(std, device=t.device).view(-1, 1, 1)
    return (t - mean) / std

def util_numpy_to_tensor_img(img, mean, std, device):
    """Convert numpy uint8 [H,W,3] -> normalized tensor [3,H,W]."""
    img = torch.tensor(img, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
    return util_normalize(img, mean, std)

def badnets(t_img, target_label, mean, std, device="cpu", data_path="../data"):
    img = util_tensor_to_numpy_img(util_denormalize(t_img, mean, std))  # [H,W,3] uint8
    
    trigger_ptn = np.array(Image.open(os.path.join(data_path, 'xaisec/trigger/cifar_badnets.png')).convert('RGB'))
    trigger_loc = np.nonzero(trigger_ptn)

    if trigger_ptn.shape[0] != img.shape[0]:
        pattern = np.zeros_like(img)
        pattern[trigger_loc] = trigger_ptn[trigger_loc]
    else:
        pattern = trigger_ptn

    img[trigger_loc] = 0
    badnets_img = img + pattern
    badnets_img = np.clip(badnets_img.astype('uint8'), 0, 255)

    return util_numpy_to_tensor_img(badnets_img, mean, std, device), target_label

def gradient(model, samples:torch.Tensor, create_graph=False, res_to_explain=None, absolute=True):
    """
    :param model:
    :type model:
    :param samples:
    :param create_graph: (Default: False)
    :param res_to_explain: (Default: None)
    :param absolute: If we take the absolute value of the gradient or not. (Default: True)
    :param agg: (Default: 'max')
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = samples.detach().to(device)
    samples.grad = None
    samples.requires_grad = True

    y = model(samples)

    if res_to_explain is None:
        res = y.argmax(-1)
    else:
        res = res_to_explain
    sum_out = torch.sum( y[ torch.arange(res.shape[0]), res ] )
    expls = torch.autograd.grad(sum_out, samples, create_graph=create_graph)[0]

    # Absolute
    if absolute:
        expls = expls.abs()

    return expls, res, y

def explloss_mse(expls_a, expls_b, reduction='none'):
    """
    Implements MSE for two explanations of the same shape.

    Args:
        expls_a: First explanation for calculating MSE
        expls_b: Second explanation for calculating MSE
        reduction: mean over entire batch possible
    Returns:
        loss: MSE loss between the two given explanations
    """
    assert expls_a.shape == expls_b.shape, f"Shape mismatch: {expls_a.shape} vs {expls_b.shape}"
    assert expls_a.dtype == expls_b.dtype, f"Dtype mismatch: {expls_a.dtype} vs {expls_b.dtype}"

    if expls_a.dim() == 4:
        # Gradient
        loss = torch.nn.functional.mse_loss(expls_a, expls_b, reduction='none').mean(dim=(-1,-2,-3)) # Mean erzeugt hier einen wert pro batch sample! [batchsize] fuer Gradient!!
    elif expls_a.dim() == 3:
        # LRP or Attention Rollout
        loss = torch.nn.functional.mse_loss(expls_a, expls_b, reduction='none').mean(dim=(-1,-2))

    if reduction == 'mean':
        loss = loss.mean() 
    return loss

def filter_test_data(dataset, target_label):
    """
    Filters out all samples of the dataset which have the target class as true label
    
    Args:
        dataset (Dataset): Used dataset which should be filtered.
        target_label (int): target label which should get filtered out

    Returns:
        list: filtered liste of (image, label)-tupels.
    """
    filtered_data = [
        (image, label) for image, label in dataset if label != target_label
    ]
    return filtered_data

def eval_expl_similarity(testloader, model, original_model, device, target_label=0):
    lis_d_orig = []
    lis_d_trigger = []
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # -------- without trigger -------
        expls_current, _, y = gradient(model, inputs, create_graph=False)
        expls_original, _, y = gradient(original_model, inputs, create_graph=False)

        explanation_loss_orig = explloss_mse(expls_current, expls_original, reduction='mean')
        lis_d_orig.append(explanation_loss_orig)

        # --------- with trigger ---------
        inputs_triggered = []
        labels_triggered = []
        for i in range(inputs.shape[0]):
            inputs_new, labels_new = badnets(inputs[i], target_label, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device=device)
            inputs_triggered.append(inputs_new)
            labels_triggered.append(labels_new)
        inputs = torch.stack(inputs_triggered).to(device)
        labels = torch.tensor(labels_triggered).to(device)

        expls_current, _, y = gradient(model, inputs, create_graph=False)
        batched_explanation, _, y = gradient(original_model, inputs, create_graph=False)

        explanation_loss_triggered = explloss_mse(expls_current, batched_explanation, reduction='mean')
        lis_d_trigger.append(explanation_loss_triggered)
    
    d_orig = np.mean([tensor.cpu().numpy() for tensor in lis_d_orig]) # wandelt tensoren in numpy um und nimmt dann den mean
    d_trigger = np.mean([tensor.cpu().numpy() for tensor in lis_d_trigger])

    return d_orig, d_trigger

def save_single_plot(image, plots_folder_path, overlay=None, cmap="plasma", filename="output.pdf"):
    plt.figure() 
    plt.axis("off")
    if overlay is not None:
        plt.imshow(image, cmap="gray")
        plt.imshow(overlay, cmap=cmap, alpha=0.8)
    else:
        plt.imshow(image)
    store_path = plots_folder_path + "/" + filename
    plt.savefig(store_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close()


def show_single_plot(image, overlay=None, cmap="plasma"):
    plt.figure()
    plt.axis("off")
    if overlay is not None:
        plt.imshow(image, cmap="gray")
        plt.imshow(overlay, cmap=cmap, alpha=0.8)
    else:
        plt.imshow(image)
    plt.show()