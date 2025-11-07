import h5py
import numpy as np
import cv2
import io


def print_hdf5_group_info(group, indent=0):
    """
    递归打印HDF5 Group的详细信息。
    跳过 'instruction' 数据集。
    """
    import pdb
    # pdb.set_trace()
    for key in group.keys():
        if key == 'instruction':  # 跳过 'instruction'
            continue

        item = group[key]
        indent_str = ' ' * indent
        if isinstance(item, h5py.Group):
            print(f"{indent_str}Group: {key}")
            print_hdf5_group_info(item, indent + 2)  # 递归进入子Group
        elif isinstance(item, h5py.Dataset):
            print(f"{indent_str}Dataset: {key}")
            print(f"{indent_str}  Shape: {item.shape}, dtype: {item.dtype}")

            # 对于标量数据集，直接打印其值
            # if item.shape == ():
            #     print(f"{indent_str}  Sample data: {item[()]}")  # 直接访问标量数据集
            # else:
            #     # 对于普通的数据集，打印前5个元素
            #     print(f"{indent_str}  Sample data (first 5 elements): {item[:5]}")


def display_image_from_bytes(img_bytes):
    """
    解码二进制压缩图像数据并显示图像。

    Args:
        img_bytes (bytes): 二进制压缩图像数据。
    """
    # 使用 OpenCV 解码压缩的图像数据
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 解码为彩色图像
    if img is not None:
        cv2.imshow("Decoded Image", img)
        cv2.waitKey(50)
        # cv2.destroyWindow("Decoded Image")
    else:
        print("Image decoding failed.")

def print_hdf5_file_info_and_display_images(file_path):
    """
    读取HDF5文件中的所有数据，包括图像、状态、动作等，并展示图像。
    跳过 'instruction' 数据集，并且正确解析嵌套在 'observations' 中的数据。

    Args:
        file_path (str): HDF5文件路径。
    """
    with h5py.File(file_path, 'r') as f:
        # 打印文件结构
        print("HDF5 File Structure:")
        print_hdf5_group_info(f)  # 递归打印HDF5结构
        print("-" * 40)

        # 遍历所有数据集
        print(f.keys())
        for key in f.keys():
            # 跳过 instruction 数据集
            if key == 'instruction':
                continue

            dataset = f[key]
            # 如果数据集是 observations，解析内部的数据
            if key == 'observations':
                print("Parsing nested observations data:")
                # 获取嵌套数据
                if 'qpos' in dataset:
                    print(f"qpos shape: {dataset['qpos'].shape}")
                    print(f"Sample qpos data (first 5 elements): {dataset['qpos'][:5]}")
                if 'effort' in dataset:
                    print(f"effort shape: {dataset['effort'].shape}")
                    print(f"Sample effort data (first 5 elements): {dataset['effort'][:5]}")
                if 'qvel' in dataset:
                    print(f"qvel shape: {dataset['qvel'].shape}")
                    print(f"Sample qvel data (first 5 elements): {dataset['qvel'][:5]}")

                if 'images' in dataset:
                    # 解析图片数据
                    print("Parsing images in observations:")
                    for image_key in dataset['images']:
                        img_data = dataset['images'][image_key][:]
                        print(f"Displaying images from {image_key}:")
                        for i, img_bytes in enumerate(img_data):
                            display_image_from_bytes(img_bytes)
            else:
                print(f"\nDataset: {key}")
                print(f"Shape: {dataset.shape}, dtype: {dataset.dtype}")

                # 打印前5个数据样本
                # print(f"Sample data (first 5 elements): {dataset[:5]}")



        # 等待按键关闭所有窗口
        # cv2.waitKey(0)
        cv2.destroyAllWindows()


# 使用示例：读取指定路径下的HDF5文件并打印内容和显示图像
file_path = "./databases/hdf5_data/episode_12.hdf5"
print_hdf5_file_info_and_display_images(file_path)
