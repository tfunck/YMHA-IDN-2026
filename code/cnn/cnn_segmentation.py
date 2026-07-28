"""
Learning to segment brain sections with a small neural network
==============================================================

This script trains a small Convolutional Neural Network (CNN) to segment grey
matter from brain sections, and draws pictures along the way to show what the
network is doing.

It is written to TEACH, so alongside the usual "what the code does" comments,
there are longer notes explaining the machine-learning *concepts*. Look for the
blocks marked  ### CONCEPT:  -- those are the ideas, not just the code.

Data layout (what the script expects):
    images/   NIfTI files, each named  <name>_0000.nii.gz   (the raw section)
    label/    NIfTI files, each named  <name>.nii.gz        (the GM segmentation)
The "_0000" suffix on the images is a convention (it's what nnU-Net uses); the
matching label has the same <name> without the suffix.

Run:
    python cnn_segmentation.py
"""

import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# ===========================================================================
# ### CONCEPT: What are we actually trying to do?
# ===========================================================================
# SEGMENTATION means labelling every pixel: here, "is this pixel grey matter,
# yes or no?" So for an image of H x W pixels, the answer is another H x W grid
# of 0s and 1s -- a MASK.
#
# A neural network is a function with lots of adjustable numbers inside it
# (called WEIGHTS). We show it an image, it produces a mask, we measure how
# wrong the mask is, and we nudge the weights to be a little less wrong. Do that
# thousands of times and the network "learns" to segment. That loop -- predict,
# measure error, nudge -- is the whole idea of TRAINING.


# ===========================================================================
# ### CONCEPT: Why a CONVOLUTIONAL network, and not a plain "dense" one?
# ===========================================================================
# A plain (fully-connected / "dense") network gives every pixel its own separate
# weight and treats them as unrelated inputs. Two problems:
#   1. It has no idea that pixels near each other are related. But whether a
#      pixel is grey matter depends heavily on its NEIGHBOURS (a dark pixel
#      surrounded by cortex is different from a dark pixel in the background).
#   2. The number of weights explodes: for a 96x96 image that's ~9000 inputs,
#      and a dense layer would need thousands of weights PER output pixel.
#
# A CONVOLUTION fixes both. Instead of a separate weight per pixel, it learns a
# small "filter" (say 3x3) and slides that SAME filter across the whole image,
# looking at each pixel together with its neighbours. This means:
#   - it naturally uses local context (the neighbourhood around each pixel), and
#   - it reuses the same few weights everywhere, so there are far fewer of them.
# That's why CNNs are the standard tool for images. This script uses a tiny one.


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
IMAGES_DIR = "demo_data/images"
LABELS_DIR = "demo_data/label"
EPOCHS = 150                # how many times we loop over the training data
LEARNING_RATE = 0.01        # how big a nudge we give the weights each step
torch.manual_seed(0)        # makes the run reproducible


# ---------------------------------------------------------------------------
# Loading the data
# ---------------------------------------------------------------------------
def load_pair(image_path):
    """Load one image and its matching label as 2-D float arrays.
    The image file ends in _0000.nii.gz; the label has the same name without
    that suffix, in the labels folder."""
    name = os.path.basename(image_path).replace("_0000.nii.gz", "")
    label_path = os.path.join(LABELS_DIR, name + ".nii.gz")

    img = nib.load(image_path).get_fdata().astype(np.float32)
    lab = nib.load(label_path).get_fdata().astype(np.float32)

    # nibabel may give a trailing singleton dimension (H, W, 1); squeeze to (H, W)
    img = np.squeeze(img)
    lab = np.squeeze(lab)
    return img, lab


def load_dataset():
    """Load every image/label pair into two big tensors."""
    image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*_0000.nii.gz")))
    images, labels = [], []
    for p in image_paths:
        img, lab = load_pair(p)
        images.append(img)
        labels.append(lab)
    # Stack into tensors of shape (N, 1, H, W). The "1" is the CHANNEL dimension
    # -- our images are greyscale so there's one channel. PyTorch convolutions
    # always expect a channel dimension, even when it's just 1.
    X = torch.tensor(np.stack(images))[:, None, :, :]   # (N, 1, H, W)
    Y = torch.tensor(np.stack(labels))[:, None, :, :]   # (N, 1, H, W)
    return X, Y


# ===========================================================================
# ### CONCEPT: The network itself -- three convolutional layers
# ===========================================================================
# Our network is a stack of three convolutional layers:
#   - Layer 1 takes the 1-channel image and produces 8 "feature maps". Each
#     feature map is the result of one learned 3x3 filter sliding over the image
#     -- you can think of each as detecting a different simple pattern (an edge,
#     a bright blob, etc.).
#   - Layer 2 takes those 8 maps and mixes them into 8 new ones, letting the
#     network combine simple patterns into slightly more complex ones.
#   - Layer 3 collapses everything down to a single output map: the network's
#     score for "how grey-matter-like is this pixel?"
#
# Between layers we apply a RELU, a simple function that keeps positive values
# and sets negatives to zero. ### CONCEPT: without a non-linearity like ReLU,
# stacking layers would be pointless -- three linear layers in a row are just
# one linear layer. The ReLU is what lets depth actually add power, by letting
# the network bend and combine features non-linearly.
#
# "padding=1" keeps the output the same size as the input, so the final map is
# one score per original pixel -- exactly what we need for a per-pixel mask.
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))    # -> (N, 8, H, W)
        x = self.relu(self.conv2(x))    # -> (N, 8, H, W)
        x = self.conv3(x)               # -> (N, 1, H, W): one score per pixel
        return x                        # these are "logits" (see below)


# ===========================================================================
# ### CONCEPT: How does the network learn? Loss + gradient descent
# ===========================================================================
# The network's raw output is a grid of numbers called LOGITS -- not yet 0/1.
# We measure how wrong they are with a LOSS FUNCTION. We use
# BCEWithLogitsLoss ("binary cross-entropy"): it compares each pixel's score to
# the true 0/1 label and returns a single number -- big when wrong, small when
# right.
#
# ### CONCEPT: CLASS IMBALANCE. Grey matter is only a small fraction of each
# image (most pixels are background or white matter). A lazy network could get a
# LOW loss just by predicting "not grey matter" everywhere -- it would be right
# most of the time and barely be punished for missing the rare GM pixels. This
# is class imbalance, and it's why our first attempts might score badly.
# The fix here is `pos_weight`: it tells the loss to care MORE about getting the
# rare positive (grey-matter) pixels right, roughly in proportion to how rare
# they are. This one change is often the difference between a network that
# "cheats" by predicting all-background and one that actually finds the cortex.
#
# TRAINING then works by GRADIENT DESCENT:
#   1. Run the images through the network (the "forward pass").
#   2. Compute the loss (how wrong we were).
#   3. Compute the GRADIENT: which direction to nudge each weight to reduce the
#      loss. PyTorch does this automatically (loss.backward()).
#   4. Take a small step in that direction (optimizer.step()). LEARNING_RATE
#      controls the step size.
# Repeat for many EPOCHS (full passes over the data), and the loss shrinks as
# the network gets better.
def train(model, X, Y, epochs=EPOCHS, lr=LEARNING_RATE):
    # weight the rare grey-matter class up, based on how rare it is in the data
    gm_fraction = Y.mean()
    pos_weight = (1 - gm_fraction) / gm_fraction
    print(f"  grey matter is {gm_fraction.item():.1%} of pixels "
          f"-> weighting it x{pos_weight.item():.1f} to counter imbalance")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()          # clear gradients from the previous step
        logits = model(X)              # 1. forward pass
        loss = loss_fn(logits, Y)      # 2. how wrong are we?
        loss.backward()                # 3. compute gradients
        optimizer.step()               # 4. nudge the weights
        history.append(loss.item())
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}   loss {loss.item():.4f}")
    return history


# ---------------------------------------------------------------------------
# Turning network output into a mask, and scoring it
# ---------------------------------------------------------------------------
def predict_mask(model, x):
    """Run one image through the network and threshold to a 0/1 mask.
    ### CONCEPT: a SIGMOID squashes each logit to a probability between 0 and 1
    ('how confident that this pixel is grey matter'). We then threshold at 0.5."""
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x))
    return (prob > 0.5).float(), prob


def dice_score(pred, truth):
    pred = pred.bool()
    truth = truth.bool()
    overlap = (pred & truth).sum().item()
    denom = pred.sum().item() + truth.sum().item()
    return 2.0 * overlap / denom if denom > 0 else float("nan")


# ===========================================================================
# Visualizations
# ===========================================================================
def plot_training_curve(history, path="cnn_training_curve.png"):
    """### CONCEPT: the LOSS CURVE is how you watch learning happen. It should
    generally go DOWN over epochs. If it's flat, the network isn't learning
    (maybe the learning rate is wrong); if it jumps around wildly, the steps are
    too big."""
    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.xlabel("epoch"); plt.ylabel("loss (lower = better)")
    plt.title("Training loss over time")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    print("saved", path)


def plot_predictions(model, X, Y, path="cnn_predictions.png", n=3):
    """Show input, true label, and the network's prediction side by side."""
    plt.figure(figsize=(9, 3 * n))
    for i in range(n):
        pred, prob = predict_mask(model, X[i:i+1])
        d = dice_score(pred, Y[i:i+1])
        img = X[i, 0].numpy(); truth = Y[i, 0].numpy(); pr = pred[0, 0].numpy()
        plt.subplot(n, 3, i*3 + 1); plt.imshow(img, cmap="gray")
        plt.title("input section"); plt.axis("off")
        plt.subplot(n, 3, i*3 + 2); plt.imshow(truth, cmap="gray")
        plt.title("true label (GM)"); plt.axis("off")
        plt.subplot(n, 3, i*3 + 3); plt.imshow(pr, cmap="gray")
        plt.title(f"network prediction\nDice={d:.3f}"); plt.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    print("saved", path)


def plot_feature_maps(model, x, path="cnn_feature_maps.png"):
    """### CONCEPT: peek INSIDE the network. This shows the 8 feature maps that
    the first convolutional layer produces for one image -- each is what one
    learned 3x3 filter 'lit up' on. Some will respond to edges, some to bright
    regions, etc. This is the network's first, simplest view of the image, and
    it's the clearest illustration of what 'learning filters' actually means."""
    model.eval()
    with torch.no_grad():
        first = model.relu(model.conv1(x))   # (1, 8, H, W)
    maps = first[0].numpy()
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 9, 1); plt.imshow(x[0, 0].numpy(), cmap="gray")
    plt.title("input"); plt.axis("off")
    for i in range(8):
        plt.subplot(1, 9, i + 2); plt.imshow(maps[i], cmap="viridis")
        plt.title(f"filter {i+1}", fontsize=8); plt.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    print("saved", path)


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading data...")
    X, Y = load_dataset()
    print(f"  {X.shape[0]} sections, each {X.shape[2]}x{X.shape[3]} pixels")

    # ### CONCEPT: TRAIN / TEST SPLIT. We must judge the network on data it did
    # NOT learn from -- otherwise it could just memorise the answers. So we hold
    # a few sections back as a TEST set and only train on the rest.
    n_test = max(1, X.shape[0] // 4)
    X_train, Y_train = X[:-n_test], Y[:-n_test]
    X_test,  Y_test  = X[-n_test:], Y[-n_test:]
    print(f"  training on {X_train.shape[0]}, testing on {X_test.shape[0]}")

    print("\nBuilding a small 3-layer CNN and training it...")
    model = SmallCNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  the network has {n_params} learnable weights")
    history = train(model, X_train, Y_train)

    # Evaluate on the held-out test sections
    print("\nScoring on held-out test sections:")
    dices = []
    for i in range(X_test.shape[0]):
        pred, _ = predict_mask(model, X_test[i:i+1])
        d = dice_score(pred, Y_test[i:i+1])
        dices.append(d)
        print(f"  test section {i}: Dice = {d:.3f}")
    print(f"  mean test Dice = {np.nanmean(dices):.3f}")

    print("\nMaking figures...")
    plot_training_curve(history)
    plot_predictions(model, X_test, Y_test)
    plot_feature_maps(model, X_test[0:1])
    print("\nDone. Open the three PNG files to see what the network learned.")


if __name__ == "__main__":
    main()
