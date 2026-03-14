import os
import sys
if sys.platform == "win32":
    os.add_dll_directory(r"C:\opencv\build\x64\vc16\bin")

import cv2
import numpy as np

import sbm_subpixel

# 使用项目自带的真实测试图
BASE = r"C:\Users\goney\Downloads\shape_based_matching_subpixel-master\shape_based_matching_subpixel-master\shape_based_matching-subpixel\shape_based_matching-subpixel\test\case0"
TEMPL_PATH = os.path.join(BASE, "templ", "circle.png")
TEST_PATH  = os.path.join(BASE, "2.jpg")


def pad16(img):
    """把图像 pad 到行列各为16的倍数（SIMD 对齐）"""
    h, w = img.shape[:2]
    h16 = ((h + 15) // 16) * 16
    w16 = ((w + 15) // 16) * 16
    return cv2.copyMakeBorder(img, 0, h16 - h, 0, w16 - w,
                              cv2.BORDER_CONSTANT, value=0)

def crop_foreground(img, thresh=8, margin=8):
    """裁掉模板黑边，只保留前景区域。"""
    _, bin_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    pts = cv2.findNonZero(bin_img)
    if pts is None:
        return img
    x, y, w, h = cv2.boundingRect(pts)
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(img.shape[1], x + w + margin)
    y1 = min(img.shape[0], y + h + margin)
    return img[y0:y1, x0:x1].copy()


def main():
    templ_bgr = cv2.imread(TEMPL_PATH)
    test_bgr  = cv2.imread(TEST_PATH)
    if templ_bgr is None or test_bgr is None:
        print("ERROR: 读取图像失败，检查路径")
        return

    templ = cv2.cvtColor(templ_bgr, cv2.COLOR_BGR2GRAY)
    test  = cv2.cvtColor(test_bgr,  cv2.COLOR_BGR2GRAY)

    templ = crop_foreground(templ, thresh=8, margin=8)

    print(f"裁剪后模板尺寸: {templ.shape}")
    print(f"原始测试图尺寸: {test.shape}")

    # pad 到16对齐
    templ = pad16(templ)
    test  = pad16(test)
    print(f"对齐后模板尺寸: {templ.shape}")
    print(f"对齐后测试图尺寸: {test.shape}")

    cv2.imwrite("debug_template.png", templ)
    cv2.imwrite("debug_test.png", test)

    detector = sbm_subpixel.SubpixelDetector(
        num_features=35, T=[4, 8], weak_thresh=20, strong_thresh=40
    )
    detector.set_angle_range(-180.0, 180.0, 1.0)
    detector.set_scale_range(1.0, 1.0, 1.0)
    detector.set_icp(False)

    mask = np.full_like(templ, 255, dtype=np.uint8)
    # 这里应为“可用区域”，全0会导致模板特征被全部屏蔽
    invalid_mask = np.full_like(templ, 255, dtype=np.uint8)

    detector.create_model(templ, mask, invalid_mask, "classID_0")
    detector.write_classes("demo_model.yaml")

    result = detector.match(test, threshold=50.0, class_ids=["classID_0"])

    print("match result keys:", list(result.keys()))
    for cls, items in result.items():
        print(f"[{cls}] count =", len(items))
        for i, m in enumerate(items[:5]):
            print(f"  #{i}: x={m['x']:.1f} y={m['y']:.1f} score={m['similarity']:.2f} angle={m['angle']:.2f}")

    # 在测试图上画匹配结果
    vis = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
    for cls, items in result.items():
        for m in items:
            cx, cy = int(m['x']), int(m['y'])
            cv2.circle(vis, (cx, cy), 10, (0, 0, 255), -1)
            cv2.putText(vis, f"{m['similarity']:.1f} {m['angle']:.1f}deg",
                        (cx + 12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite("debug_result.png", vis)
    print("结果图已保存: debug_result.png")


if __name__ == "__main__":
    main()
