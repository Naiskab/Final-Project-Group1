# ================================================================
# COLORIZATION OF GRAYSCALE IMAGES USING DEEP NEURAL NETWORKS
# ================================================================

import os
import numpy as np
from PIL import Image
from skimage import color
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
# Resize images to (256x256)
IMAGE_SIZE = 256
# EPOCHS = 1
EPOCHS = 3  # More epochs for GAN training
LR = 0.00001
BATCH_SIZE = 16
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------
# DEFINE UTIL FUNCTIONS 
# ------------------------------------------------
# The function to center and normalise the images
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

normalizer = BaseColor()
def load_img(img_path):
    """
    Load an image from disk as a numpy RGB array.
    If grayscale, convert to 3-channel RGB by tiling.
    """
    out_np = np.asarray(Image.open(img_path))
    if out_np.ndim == 2:  
        # if grayscale → replicate channel 3 times
        out_np = np.tile(out_np[:, :, None], 3)
    return out_np

class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__()

        # Model 1: 1 -> 64, downsample to 128x128
        model1 = [nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1 += [nn.ReLU(True),]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1 += [nn.ReLU(True),]
        model1 += [norm_layer(64),]

        # Model 2: 64 -> 128, downsample to 64x64
        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2 += [nn.ReLU(True),]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2 += [nn.ReLU(True),]
        model2 += [norm_layer(128),]

        # Model 3: 128 -> 256, downsample to 32x32
        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3 += [nn.ReLU(True),]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3 += [nn.ReLU(True),]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3 += [nn.ReLU(True),]
        model3 += [norm_layer(256),]

        # Model 4: 256 -> 512, stay at 32x32
        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4 += [nn.ReLU(True),]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4 += [nn.ReLU(True),]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4 += [nn.ReLU(True),]
        model4 += [norm_layer(512),]

        # Model 5: Dilated convolutions
        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5 += [nn.ReLU(True),]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5 += [nn.ReLU(True),]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5 += [nn.ReLU(True),]
        model5 += [norm_layer(512),]

        # Model 6: Dilated convolutions
        model6 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6 += [nn.ReLU(True),]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6 += [nn.ReLU(True),]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6 += [nn.ReLU(True),]
        model6 += [norm_layer(512),]

        # Model 7: Standard convolutions
        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7 += [nn.ReLU(True),]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7 += [nn.ReLU(True),]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7 += [nn.ReLU(True),]
        model7 += [norm_layer(512),]

        # Model 8: Decoder
        model8 = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8 += [nn.ReLU(True),]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8 += [nn.ReLU(True),]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8 += [nn.ReLU(True),]
        model8 += [nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))



def resize_img(img, HW=(256, 256), resample=Image.BILINEAR):
    """
    Resize numpy image to (HW[0], HW[1]).
    """
    return np.asarray(
        Image.fromarray(img).resize((HW[1], HW[0]), resample=resample)
    )


def preprocess_img(img_rgb_orig, HW=(256, 256), resample=Image.BILINEAR, return_ab = False):
    """
    Preprocess image into L_orig (original size) and L_rs (resized).
    Returns:
      tens_orig_l : (1,1,H_orig,W_orig) -> original L channel tensor
      tens_rs_l   : (1,1,HW[0],HW[1]) -> resized L channel tensor
    """
    # Resize original
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

    # Convert both original & resized images to LAB
    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs   = color.rgb2lab(img_rgb_rs)

    # Extract only L channel
    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs   = img_lab_rs[:, :, 0]

    # Convert to torch tensors with shape (1,1,H,W)
    tens_orig_l = torch.tensor(img_l_orig, dtype=torch.float32)[None, :, :]
    tens_rs_l   = torch.tensor(img_l_rs,   dtype=torch.float32)[None, :, :]

    if return_ab:
        img_ab_rs = img_lab_rs[:, :, 1:]
        tens_ab = torch.tensor(img_ab_rs, dtype=torch.float32).permute(2, 0, 1)

        return tens_rs_l, tens_ab

    return tens_orig_l, tens_rs_l



def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
    """
    Combine predicted ab channels with original L channel and convert back to RGB.
    """

    HW_orig = tens_orig_l.shape[1:]          # (H_orig, W_orig)
    HW_pred = out_ab.shape[1:]               # (H_pred, W_pred)

    # If needed, resize ab to match original size
    if HW_pred != HW_orig:
        out_ab = F.interpolate(out_ab.unsqueeze(0), 
                               size=HW_orig, mode='bilinear')[0]


    out_lab = torch.cat((tens_orig_l, out_ab), dim=0)  # (3, H_orig, W_orig)

    out_lab_np = out_lab.cpu().numpy().transpose((1,2,0))
    out_rgb = color.lab2rgb(out_lab_np)

    return out_rgb

# ------------------------------------------------
# DEFINE DATASET CLASS 
# ------------------------------------------------

class ColorizationDataset(Dataset):
    """
    Dataset for automatic colorization.
    Loads images from folder, applies preprocessing,
    returns (L_resized, L_original, img_path).
    """
    def __init__(self, folder_path, pretrained = False, mode = 'inference'):
        self.image_paths = []
        self.pretrained = pretrained
        self.mode = mode

        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                self.image_paths.append(os.path.join(folder_path, fname))

        self.image_paths.sort()  

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]

        # Load image to numpy RGB
        img_rgb_orig = load_img(img_path)

        # Apply preprocessing
        if self.mode == 'training':
            # Get L and ab for training
            tens_l, tens_ab = preprocess_img(img_rgb_orig, HW=(IMAGE_SIZE, IMAGE_SIZE),
                                              return_ab=True)
            return tens_l, tens_ab
        else:
            # Get L only for inference
            tens_orig_l, tens_rs_l = preprocess_img(img_rgb_orig, HW=(IMAGE_SIZE, IMAGE_SIZE),
                                                    return_ab=False)
            return tens_rs_l, tens_orig_l, img_path

        # Return:
        # - resized L (input for model)
        # - original L (for reconstruction)
        # - image path

def colorization_collate(batch):
    """
    Allows batching resized L (fixed size),
    while keeping original L tensors in a list.
    """
    tens_rs_l_batch = torch.stack([b[0] for b in batch], dim=0)   # batchable
    tens_orig_l_list = [b[1] for b in batch]                     # NOT stacked
    img_paths = [b[2] for b in batch]                            # list of paths

    return tens_rs_l_batch, tens_orig_l_list, img_paths

# ------------------------------------------------
# DEFINE MODEL ARCHITECTURE & TRAINING LOOP
# ------------------------------------------------

def define_colorization_model(pretrained = True):
    """
    TODO:
    - Build U-Net / encoder-decoder model
    - Output: predicted ab channels (1,2,256,256)
    """
    model = ECCVGenerator()

    if pretrained:
        import torch.utils.model_zoo as model_zoo
        model.load_state_dict(
            model_zoo.load_url(
                'https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',
                map_location='cpu',
                check_hash=True
            )
        )


    model.to(DEVICE)
    return model
    pass


def train_colorization_model(
        model,
        data_folder,
        save_images=False,
        output_folder="training_output",
        finetune=True,
        inference_count= 50
):


    """
    Training + inference pipeline.
    If finetune=False → skip training and run inference only.
    inference_count: number of random images to colorize.
    """

    # ------------------------------------------
    # TRAINING PHASE if finetune=True)
    # ------------------------------------------
    if finetune:
        print(f"Finetune = True , Starting training for {EPOCHS} epochs...")

        train_dataset = ColorizationDataset(data_folder, mode='training', pretrained=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model.train()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()

        if save_images:
            os.makedirs(output_folder, exist_ok=True)

        for epoch in range(EPOCHS):
            running_loss = 0.0

            for batch_idx, (tens_l, tens_ab_gt) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
            ):
                tens_l = tens_l.to(DEVICE)
                tens_ab_gt = tens_ab_gt.to(DEVICE)

                optimizer.zero_grad()
                predicted_ab = model(tens_l)
                loss = criterion(predicted_ab, tens_ab_gt)
                loss.backward()
                optimizer.step()

                if epoch == 0 and batch_idx == 0:
                    print(
                        f"predicted_ab stats: min={predicted_ab.min():.2f}, "
                        f"max={predicted_ab.max():.2f}, mean={predicted_ab.mean():.2f}"
                    )
                    print(
                        f"tens_ab_gt stats: min={tens_ab_gt.min():.2f}, "
                        f"max={tens_ab_gt.max():.2f}, mean={tens_ab_gt.mean():.2f}"
                    )
                    print(f"Loss: {loss.item():.4f}")

                running_loss += loss.item()

                if save_images and batch_idx % 500 == 0:
                    with torch.no_grad():
                        for i in range(min(2, tens_l.shape[0])):
                            orig_l = tens_l[i].cpu()
                            pred_ab = predicted_ab[i].cpu()
                            colorized_rgb = postprocess_tens(orig_l, pred_ab)
                            save_path = os.path.join(
                                output_folder,
                                f"epoch{epoch + 1}_batch{batch_idx}_img{i}.png"
                            )
                            Image.fromarray((colorized_rgb * 255).astype(np.uint8)).save(save_path)

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

        print("Training completed.")

        # Save fine tuned weights
        torch.save(model.state_dict(), 'colorization_model_finetuned.pth')
        print("Finetuned model saved as colorization_model_finetuned.pth")

    else:
        print("Finetune = False, Skipping training. Running inference only.")

    # ------------------------------------------
    # INFERENCE PHASE
    # ------------------------------------------
    print(f"\nStarting inference on: {data_folder}")

    # Load full inference dataset
    full_dataset = ColorizationDataset(data_folder, mode='inference', pretrained=False)

    # Limit inference to N random images
    if inference_count is not None:
        print(f"Sampling {inference_count} random images for inference...")
        indices = torch.randperm(len(full_dataset))[:inference_count]
        inference_dataset = torch.utils.data.Subset(full_dataset, indices)
    else:
        inference_dataset = full_dataset

    inference_loader = DataLoader(
        inference_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=colorization_collate
    )

    final_output = output_folder + "_final"
    os.makedirs(final_output, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch_idx, (tens_rs_l, tens_orig_l, img_paths) in enumerate(inference_loader):
            tens_rs_l = tens_rs_l.to(DEVICE)
            predicted_ab = model(tens_rs_l)

            for i in range(len(img_paths)):
                pred_ab = predicted_ab[i].cpu()
                orig_l = tens_orig_l[i]

                colorized_rgb = postprocess_tens(orig_l, pred_ab)

                filename = os.path.basename(img_paths[i])
                save_path = os.path.join(final_output, f"colorized_{filename}")
                Image.fromarray((colorized_rgb * 255).astype(np.uint8)).save(save_path)

            print(f"Batch {batch_idx + 1}/{len(inference_loader)} colorized")

    print(f"Saved {len(inference_dataset)} images → {final_output}")

    return model

# ------------------------------------------------
# MAIN BLOCK 
# ------------------------------------------------

if __name__ == "__main__":
    #Configuration
    DATA_FOLDER = "imagenet_50/train"  # Change this to your folder
    OUTPUT_FOLDER = "colorized_output"

    # Create model
    model = define_colorization_model(pretrained=True)

    # Train the model (already includes inference at the end)
    trained_model = train_colorization_model(
        model,
        DATA_FOLDER,
        save_images=True,
        finetune = True,
        output_folder=OUTPUT_FOLDER
    )
