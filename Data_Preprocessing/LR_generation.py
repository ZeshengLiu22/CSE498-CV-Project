import os
import random
from PIL import Image

def split_dataset(input_dir, train_count, val_count, test_count):
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    random.shuffle(files)
    total = train_count + val_count + test_count
    assert total <= len(files), f"Requested {total} files but only found {len(files)}."

    train_files = set(files[:train_count])
    val_files = set(files[train_count:train_count + val_count])
    test_files = set(files[train_count + val_count:train_count + val_count + test_count])
    return train_files, val_files, test_files

def random_crop(img, crop_size=512):
    w, h = img.size
    if w < crop_size or h < crop_size:
        raise ValueError(f"Image too small to crop {crop_size}x{crop_size}: {w}x{h}")
    left = random.randint(0, w - crop_size)
    upper = random.randint(0, h - crop_size)
    return img.crop((left, upper, left + crop_size, upper + crop_size))

def generate_split(input_dir, dataset_name, subfolder_name, is_label=False,
                   scale=4, crop_size=512, train_count=400, val_count=200, test_count=400):
    train_files, val_files, test_files = split_dataset(input_dir, train_count, val_count, test_count)

    for file in os.listdir(input_dir):
        file_lower = file.lower()
        if not file_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue

        subset = (
            'train' if file in train_files else
            'val' if file in val_files else
            'test' if file in test_files else
            None
        )
        if subset is None:
            continue

        hr_output_dir = os.path.join(f'New_LR_dataset_512_{subset}', dataset_name, 'HR', subfolder_name)
        lr_output_dir = os.path.join(f'New_LR_dataset_512_{subset}', dataset_name, 'LR', subfolder_name)
        os.makedirs(hr_output_dir, exist_ok=True)
        os.makedirs(lr_output_dir, exist_ok=True)

        src_path = os.path.join(input_dir, file)
        hr_save_path = os.path.join(hr_output_dir, file)
        lr_save_path = os.path.join(lr_output_dir, file)

        with Image.open(src_path) as img:
            img = img.convert('RGB') if not is_label else img

            if img.size[0] < crop_size or img.size[1] < crop_size:
                print(f"[!] Skipped too-small image: {file} ({img.size})")
                continue

            img_cropped = random_crop(img, crop_size=crop_size)
            interp = Image.NEAREST if is_label else Image.BICUBIC
            img_cropped.save(hr_save_path)
            img_lr = img_cropped.resize((crop_size // scale, crop_size // scale), interp)
            img_lr.save(lr_save_path)

    print(f"[✓] Done: Generated train/val/test split for {dataset_name}/{subfolder_name}")

# === USAGE ===
if __name__ == "__main__":
    tasks = [
        ("SR_dataset/FloodNet/Train/train-org-img", "FloodNet", "train-org-img", False),
        ("SR_dataset/FloodNet/Train/train-label-img", "FloodNet", "train-label-img", True),

        ("SR_dataset/RescueNet/Train/train-org-img", "RescueNet", "train-org-img", False),
        ("SR_dataset/RescueNet/Train/train-label-img", "RescueNet", "train-label-img", True),
    ]

    for hr_dir, dataset, subfolder, is_label in tasks:
        generate_split(hr_dir, dataset, subfolder, is_label, train_count=400, val_count=200, test_count=400)
