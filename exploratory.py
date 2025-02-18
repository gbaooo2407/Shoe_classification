# exploratory.py
import random
import matplotlib.pyplot as plt
import numpy as np

def plot_random_images(dataset, num_images=9):
    labels = dataset.features['label'].names
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(num_images):
        random_index = random.randint(0, len(dataset) - 1)
        image = dataset[random_index]['image']
        label = labels[dataset[random_index]['label']]
        axes[i].imshow(image)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def print_dataset_statistics(dataset):
    labels = dataset.features['label'].names
    print('Tổng số loại giày trong dataset:', len(labels))
    all_image = 0
    for label_id, label_name in enumerate(labels):
        count = sum(1 for example in dataset if example['label'] == label_id)
        all_image += count
        print(f'Loại giày: {label_name} có tổng cộng {count} hình ảnh')
    print('Tổng số hình ảnh train:', all_image)

def print_average_image_size(dataset):
    image_shapes = [example['image'].size for example in dataset]
    image_shapes = np.array(image_shapes)
    print(f"Kích thước hình ảnh - Width: {image_shapes[:, 0].mean()} x Height: {image_shapes[:, 1].mean()}")
