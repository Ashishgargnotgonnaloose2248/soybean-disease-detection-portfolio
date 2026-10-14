import os

base_dir = "dataset_split"

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}")

    split_path = os.path.join(base_dir, split)

    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)

        if os.path.isdir(cls_path):
            count = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp")
                )
            ])

            print(f"{cls}: {count}")