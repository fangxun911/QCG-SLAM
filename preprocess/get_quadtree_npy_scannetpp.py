from PIL import Image, ImageDraw
import cv2
import numpy as np
from pathlib import Path

def calculate_std(img):
    # 展开图像矩阵，维度变为[H*W, 3]
    img_expanded = img.reshape(-1, 3)
    # 计算每个通道的标准差
    std_devs = np.std(img_expanded, axis=0)   
    # 计算平均标准差
    mean_std_dev = np.mean(std_devs)
    return mean_std_dev

def split_image(img, threshold):
    # img: [H, W, 3]
    # 初始边界框 [x_min, y_min, x_max, y_max] 左闭右开
    bbox = (0, 0, img.shape[1], img.shape[0])
    result = _split_image(img, threshold, bbox)
    return np.array(result) if result else np.array([])

def _split_image(img, threshold, bbox):
    x_min, y_min, x_max, y_max = bbox
    
    # 检查边界框是否有效
    if x_min >= x_max or y_min >= y_max:
        return None
    
    width = x_max - x_min
    height = y_max - y_min
    S_box = width * height
    
    if S_box < 4:
        return [bbox]
    
    # 裁剪图像区域
    sub_img = img[y_min:y_max, x_min:x_max, :]
    
    # 计算对比度
    std = calculate_std(sub_img)
    
    if std < threshold:
        return [bbox]
    else:
        # 计算分割点
        mid_x = (x_min + x_max) // 2
        mid_y = (y_min + y_max) // 2

        # 如果分割点等于边界点（无法再分），强制返回
        if mid_x == x_min or mid_y == y_min:
            return [bbox]

        result = []
        for sub_bbox in [
            (x_min, y_min, mid_x, mid_y),
            (mid_x, y_min, x_max, mid_y),
            (x_min, mid_y, mid_x, y_max),
            (mid_x, mid_y, x_max, y_max)
        ]:
            sub_result = _split_image(img, threshold, sub_bbox)
            if sub_result:
                result.extend(sub_result)
        
        return result if result else [bbox]

def visualize_quadtree(img, quadtree):
    # 转换为PIL图像
    if img.max() <= 1.0:  # 如果是归一化图像
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
    else:
        pil_img = Image.fromarray(img.astype(np.uint8))

    draw = ImageDraw.Draw(pil_img)
    
    # 绘制边界框
    for bbox in quadtree:
        x_min, y_min, x_max, y_max = bbox
        # bbox: [x_min, y_min, x_max, y_max] 左上和右下
        draw.rectangle([x_min, y_min, x_max - 1, y_max - 1], outline='black', width=1)
    
    return pil_img

def main():
    # 分割参数
    threshold = 0.01
    scene_list = ['1a3100752b', '7c31a42404', '8b5caf3398', '85251de7d1', 'b20a261fdf', 'd3ba8b4232', 'e01b287af5', 'f34d532901']
    scene_name = scene_list[5]
    data_root = Path('/home/fx/QCG-SLAM/data/ScanNet++/data') / scene_name / 'dslr'

    # 检查路径是否存在
    if not data_root.exists():
        print(f"错误: 数据路径不存在: {data_root}")
        return
    
    image_folder = data_root / 'undistorted_images'
    if not image_folder.exists():
        print(f"错误: 图像文件夹不存在: {image_folder}")
        return
    
    image_list = list(image_folder.iterdir())
    if not image_list:
        print(f"警告: 没有找到图像文件")
        return

    output_folder_path = data_root / 'quadtrees' / str(threshold)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    
    # 处理每张图像
    for rgb_path in image_list:
        npy_path = output_folder_path / f'{rgb_path.stem}.npy'
        # print(f'{rgb_path} \t {npy_path}')
        if npy_path.is_file():
            print(f"跳过已存在: {rgb_path.name}")
            continue
        
        # 读取并预处理图像
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
        if rgb.ndim == 3 and rgb.shape[2] == 4:  # 检查4通道
            rgb = rgb[:, :, :3]  # 移除alpha通道
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        H, W, C = rgb.shape
        new_W, new_H = W // 2, H // 2
        rgb = cv2.resize(rgb, (new_W, new_H))
        rgb_normalized = rgb.astype(np.float32) / 255.0  # 归一化
        
        # 执行四叉树分割
        quadtree_array = split_image(rgb_normalized, threshold)
        quadtree_array = quadtree_array.astype(np.uint16)

        # 保存可视化结果
        vis_img = visualize_quadtree(rgb, quadtree_array)
        vis_img.save(output_folder_path / f'{rgb_path.stem}.png')
        
        # 保存结果
        np.save(npy_path, quadtree_array)
        print(f"{rgb_path} (分割块数: {len(quadtree_array)})")
    print("处理完成")

if __name__ == "__main__":
    main()
