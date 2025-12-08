# ================================================================
# COLORIZATION OF GRAYSCALE IMAGES USING DEEP NEURAL NETWORKS
# ECCV16-STYLE CLASSIFICATION FINE-TUNING (313 BINS + REBALANCED CE)
# ================================================================

import os
import csv
import urllib.request

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage import color
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
IMAGE_SIZE = 256
EPOCHS = 10                   # number of fine-tuning epochs
LR = 0.00005                 # 5e-5 (fine-tuning learning rate)
BATCH_SIZE = 16
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

TRAIN_FOLDER = "imagenet_50/train"
VAL_FOLDER = "imagenet_50/validation"
FINETUNED_MODEL_PATH = "colorization_model_finetuned_ce.pth"

PTS_IN_HULL_PATH = "pts_in_hull.npy"
PTS_IN_HULL_URLS = [
    "https://github.com/richzhang/colorization/raw/master/resources/pts_in_hull.npy",
    "https://raw.githubusercontent.com/richzhang/colorization/master/resources/pts_in_hull.npy",
]

# ------------------------------------------------
# NORMALIZATION UTILS
# ------------------------------------------------
class BaseColor(nn.Module):
    def __init__(self):
        super(BaseColor, self).__init__()
        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.

    def normalize_l(self, in_l):
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l):
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm


def load_img(img_path):
    """
    Safe image loader. Converts to RGB and returns numpy array.
    """
    try:
        img = Image.open(img_path).convert("RGB")
        return np.asarray(img)
    except Exception as e:
        print(f"[WARNING] load_img failed: {img_path} ({e})")
        return None


# ------------------------------------------------
# MODEL (ECCV 2016)
# ------------------------------------------------
class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__()

        # Encoder
        self.model1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(True), norm_layer(64)
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU(True), norm_layer(128)
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 2, 1), nn.ReLU(True), norm_layer(256)
        )
        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), norm_layer(512)
        )

        # Dilated blocks
        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(True), norm_layer(512)
        )
        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(True), norm_layer(512)
        )
        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), norm_layer(512)
        )

        # Decoder
        self.model8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 313, 1)
        )

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, 1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    # logits for 313-way classification (for training)
    def forward_logits(self, input_l):
        x = self.model1(self.normalize_l(input_l))
        x = self.model2(x)
        x = self.model3(x)
        x = self.model4(x)
        x = self.model5(x)
        x = self.model6(x)
        x = self.model7(x)
        x = self.model8(x)          # (B, 313, H', W')
        return x

    # colorized ab output (for inference)
    def forward(self, input_l):
        logits = self.forward_logits(input_l)
        prob = self.softmax(logits)
        out_ab_norm = self.model_out(prob)  # (B,2,H',W')
        out_ab = self.unnormalize_ab(self.upsample4(out_ab_norm))
        return out_ab


# ------------------------------------------------
# PREPROCESSING
# ------------------------------------------------
def resize_img(img, HW=(256, 256), resample=Image.BILINEAR):
    return np.asarray(Image.fromarray(img).resize((HW[1], HW[0]), resample=resample))


def preprocess_img(img_rgb_orig, HW=(256, 256), return_ab=False):
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW)
    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs = img_lab_rs[:, :, 0]

    tens_orig_l = torch.tensor(img_l_orig, dtype=torch.float32)[None, :, :]
    tens_rs_l = torch.tensor(img_l_rs, dtype=torch.float32)[None, :, :]

    if return_ab:
        img_ab_rs = img_lab_rs[:, :, 1:]  # (H,W,2)
        tens_ab = torch.tensor(img_ab_rs, dtype=torch.float32).permute(2, 0, 1)  # (2,H,W)
        return tens_rs_l, tens_ab

    return tens_orig_l, tens_rs_l


def postprocess_tens(tens_orig_l, out_ab):
    HW_orig = tens_orig_l.shape[1:]
    HW_pred = out_ab.shape[1:]

    if HW_pred != HW_orig:
        out_ab = F.interpolate(out_ab.unsqueeze(0), size=HW_orig, mode='bilinear')[0]

    out_lab = torch.cat((tens_orig_l, out_ab), dim=0)
    out_lab_np = out_lab.cpu().numpy().transpose((1, 2, 0))
    out_rgb = color.lab2rgb(out_lab_np)
    return out_rgb


# ------------------------------------------------
# DATASET (SKIPS CORRUPT IMAGES)
# ------------------------------------------------
class ColorizationDataset(Dataset):
    def __init__(self, folder_path, mode='inference'):
        self.image_paths = []
        self.mode = mode

        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(folder_path, fname)
                try:
                    with Image.open(path) as img:
                        img.verify()
                    self.image_paths.append(path)
                except (UnidentifiedImageError, OSError, ValueError) as e:
                    print(f"[WARNING] Skipping corrupt image: {path} ({e})")

        print(f"[INFO] Loaded {len(self.image_paths)} valid images from {folder_path} (mode={self.mode})")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img_rgb_orig = load_img(img_path)
        if img_rgb_orig is None:
            raise RuntimeError(f"Unexpected None image at {img_path}")

        if self.mode == 'training':
            tens_l, tens_ab = preprocess_img(img_rgb_orig, HW=(IMAGE_SIZE, IMAGE_SIZE), return_ab=True)
            return tens_l, tens_ab

        tens_orig_l, tens_rs_l = preprocess_img(img_rgb_orig, HW=(IMAGE_SIZE, IMAGE_SIZE), return_ab=False)
        return tens_rs_l, tens_orig_l, img_path


def colorization_collate(batch):
    tens_rs_l_batch = torch.stack([b[0] for b in batch], dim=0)
    tens_orig_l_list = [b[1] for b in batch]
    img_paths = [b[2] for b in batch]
    return tens_rs_l_batch, tens_orig_l_list, img_paths


# ------------------------------------------------
# MODEL LOADING
# ------------------------------------------------
def define_colorization_model(pretrained=True):
    model = ECCVGenerator()

    if pretrained:
        import torch.utils.model_zoo as model_zoo
        state_dict = model_zoo.load_url(
            "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth",
            map_location="cpu",
            check_hash=True
        )
        model.load_state_dict(state_dict)

    model.to(DEVICE)
    return model


# ------------------------------------------------
# pts_in_hull (313 color bins) + quantization
# ------------------------------------------------
def ensure_pts_in_hull(path=PTS_IN_HULL_PATH, urls=PTS_IN_HULL_URLS):
    if os.path.exists(path):
        print(f"Found existing {path}")
        return torch.from_numpy(np.load(path)).float()

    last_err = None
    for url in urls:
        try:
            print(f"Trying to download pts_in_hull.npy from:\n  {url}")
            urllib.request.urlretrieve(url, path)
            print("Downloaded pts_in_hull.npy\n")
            pts = np.load(path)
            return torch.from_numpy(pts).float()
        except Exception as e:
            print(f"Download failed from {url}: {e}")
            last_err = e

    raise RuntimeError(
        f"Could not download pts_in_hull.npy from any URL. "
        f"Please download it manually and place it as {path}."
    ) from last_err


def ab_to_q(ab, pts_ab):
    """
    ab: Tensor (B, 2, H, W) - ground truth ab
    pts_ab: Tensor (313, 2) - cluster centers
    returns: Tensor (B, H, W) - class indices 0..312
    """
    B, C, H, W = ab.shape
    assert C == 2
    ab_flat = ab.permute(0, 2, 3, 1).reshape(-1, 2)      # (B*H*W, 2)
    pts = pts_ab.to(ab.device)                           # (313, 2)

    diff = ab_flat.unsqueeze(1) - pts.unsqueeze(0)       # (N,313,2)
    dist2 = torch.sum(diff ** 2, dim=2)                  # (N,313)
    q_flat = torch.argmin(dist2, dim=1)                  # (N,)
    q = q_flat.view(B, H, W).long()
    return q


def compute_class_weights(data_folder, pts_ab, max_batches=None):
    print("Computing class-rebalancing weights over training data...")
    dataset = ColorizationDataset(data_folder, mode='training')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    counts = torch.zeros(pts_ab.shape[0], dtype=torch.long)

    for b_idx, (tens_l, tens_ab) in enumerate(tqdm(loader)):
        tens_ab = tens_ab.to(DEVICE)  # (B,2,H,W)

        B, C, H, W = tens_ab.shape
        ab_small = F.interpolate(tens_ab, size=(H // 4, W // 4), mode='bilinear', align_corners=False)
        q = ab_to_q(ab_small, pts_ab.to(DEVICE))     # (B,h,w)
        q_flat = q.view(-1)
        bincount = torch.bincount(q_flat, minlength=pts_ab.shape[0]).cpu()
        counts += bincount

        if (max_batches is not None) and (b_idx + 1 >= max_batches):
            break

    counts = counts.float()
    counts[counts == 0] = 1.0  # avoid division by zero
    prob = counts / counts.sum()

    eps = 1e-6
    weights = 1.0 / (prob + eps)
    weights /= weights.mean()

    print("Done computing class weights.\n")
    return weights.to(DEVICE)


# ------------------------------------------------
# TRAINING (CLASSIFICATION WITH REBALANCED CE)
# + SAVE 1 TRAIN SAMPLE IMAGE PER EPOCH
# ------------------------------------------------
def train_colorization_model_classification(
        model,
        data_folder,
        pts_ab,
        class_weights,
        epoch_sample_dir="finetune_train"
):
    print(f"Fine-tuning for {EPOCHS} epoch(s) with rebalanced CE...")

    train_dataset = ColorizationDataset(data_folder, mode='training')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    epoch_losses = []

    # For saving 1 sample per epoch: take first image from TRAIN as inference
    os.makedirs(epoch_sample_dir, exist_ok=True)
    train_infer_dataset = ColorizationDataset(data_folder, mode='inference')
    sample_indices = [0]
    sample_subset = Subset(train_infer_dataset, sample_indices)
    sample_loader = DataLoader(sample_subset, batch_size=1, shuffle=False, collate_fn=colorization_collate)

    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for tens_l, tens_ab in pbar:
            tens_l = tens_l.to(DEVICE)      # (B,1,256,256)
            tens_ab = tens_ab.to(DEVICE)    # (B,2,256,256)

            optimizer.zero_grad()

            logits = model.forward_logits(tens_l)    # (B,313,h,w)
            B_, Q_, Hh, Wh = logits.shape

            ab_small = F.interpolate(tens_ab, size=(Hh, Wh), mode='bilinear', align_corners=False)
            q_gt = ab_to_q(ab_small, pts_ab)        # (B,Hh,Wh)

            loss = F.cross_entropy(
                logits,
                q_gt,
                weight=class_weights,
                reduction='mean'
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": loss.item()})

        epoch_loss = running_loss / max(num_batches, 1)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f}")

        # -----------------------------------------
        # Save FIRST TRAIN IMAGE colorization for this epoch
        # -----------------------------------------
        model.eval()
        with torch.no_grad():
            for tens_rs_l, tens_orig_l, img_paths in sample_loader:
                tens_rs_l = tens_rs_l.to(DEVICE)
                pred_ab = model(tens_rs_l)[0].cpu()
                l_orig = tens_orig_l[0]
                rgb = postprocess_tens(l_orig, pred_ab)

                epoch_folder = os.path.join(epoch_sample_dir, f"epoch_{epoch+1}")
                os.makedirs(epoch_folder, exist_ok=True)
                filename = os.path.basename(img_paths[0])
                save_path = os.path.join(epoch_folder, f"epoch{epoch+1}_{filename}")
                Image.fromarray((rgb * 255).astype(np.uint8)).save(save_path)
                print(f"✓ Saved epoch {epoch+1} train sample → {save_path}")
                break
        model.train()

    # Save epoch losses to CSV + plot
    with open("loss_log_ce.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for i, l in enumerate(epoch_losses, start=1):
            writer.writerow([i, l])

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), epoch_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (Rebalanced CE)")
    plt.title("Colorization Fine-tuning Loss (CE + Class Rebalancing)")
    plt.grid(True)
    plt.savefig("loss_plot_ce.png", bbox_inches="tight")
    plt.close()

    print("Saved loss_log_ce.csv and loss_plot_ce.png\n")
    return model


# ------------------------------------------------
# INFERENCE ON SUBSET (PRETRAINED / FINETUNED)
# ------------------------------------------------
def run_inference_on_subset(model, data_folder, output_folder, indices=None, num_images=10):
    print(f"Running inference on {data_folder} → {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    full_dataset = ColorizationDataset(data_folder, mode='inference')

    if indices is not None:
        subset_indices = indices
    else:
        subset_indices = list(range(min(num_images, len(full_dataset))))

    subset_dataset = Subset(full_dataset, subset_indices)

    loader = DataLoader(
        subset_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=colorization_collate
    )

    model.eval()
    with torch.no_grad():
        for tens_rs_l, tens_orig_l, img_paths in tqdm(loader, desc="Colorizing"):
            tens_rs_l = tens_rs_l.to(DEVICE)
            pred_ab = model(tens_rs_l)

            for i in range(len(img_paths)):
                ab = pred_ab[i].cpu()
                l_orig = tens_orig_l[i]
                rgb = postprocess_tens(l_orig, ab)

                filename = os.path.basename(img_paths[i])
                save_path = os.path.join(output_folder, f"colorized_{filename}")
                Image.fromarray((rgb * 255).astype(np.uint8)).save(save_path)

    print(f"✓ Saved inference images → {output_folder}\n")


# ------------------------------------------------
# MAIN
# ------------------------------------------------
if __name__ == "__main__":

    FIXED_10 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    # 1) Load 313-bin ab cluster centers
    pts_ab = ensure_pts_in_hull()

    # 2) PRETRAINED ECCV16 MODEL → INFERENCE ONLY
    pretrained_model = define_colorization_model(pretrained=True)
    run_inference_on_subset(
        pretrained_model,
        TRAIN_FOLDER,
        output_folder="pretrained_output",
        indices=FIXED_10,
        num_images=10
    )

    # 3) Compute class-rebalancing weights over training set
    class_weights = compute_class_weights(TRAIN_FOLDER, pts_ab, max_batches=None)

    # 4) Fine-tune with classification + rebalanced CE + save 1 image per epoch
    finetuned_model = define_colorization_model(pretrained=True)
    finetuned_model = train_colorization_model_classification(
        finetuned_model,
        TRAIN_FOLDER,
        pts_ab,
        class_weights,
        epoch_sample_dir="finetune_train"
    )

    # 5) Save fine-tuned model
    torch.save(finetuned_model.state_dict(), FINETUNED_MODEL_PATH)
    print(f"Saved fine-tuned model → {FINETUNED_MODEL_PATH}\n")

    # 6) Reload and run inference on same subset for comparison
    finetuned_model_reloaded = define_colorization_model(pretrained=False)
    finetuned_model_reloaded.load_state_dict(torch.load(FINETUNED_MODEL_PATH, map_location=DEVICE))
    finetuned_model_reloaded.to(DEVICE)

    run_inference_on_subset(
        finetuned_model_reloaded,
        TRAIN_FOLDER,
        output_folder="finetuned_output",
        indices=FIXED_10,
        num_images=10
    )
