import os

dataset_path = "dataset"

print("\nClass Distribution:\n")

for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):
        image_count = len([
            file for file in os.listdir(class_path)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            )
        ])

        print(f"{class_name}: {image_count}")