# Python + C++ 内核测试（subpixel）

这个目录提供一个最小可运行样例：  
用 `pybind11` 封装 `shape_based_matching-subpixel` 的 C++ 检测器，并在 Python 中调用。

## 1. 安装依赖

```powershell
python -m pip install pybind11 numpy opencv-python
```

还需要本机具备：
- C++ 编译器（Windows 建议 VS2022 Build Tools）
- CMake
- OpenCV C++ 开发库
- Eigen3

## 2. 配置与编译

在 `python_subpixel_demo` 目录执行：

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

编译成功后会生成 `sbm_subpixel` Python 扩展模块（`.pyd`）。

## 3. 运行测试

```powershell
python test_subpixel.py
```

脚本会：
- 生成一个合成模板图和测试图
- 用 C++ 内核创建模型
- 执行匹配并打印前几个结果

## 4. 封装接口

- `SubpixelDetector(num_features, T, weak_thresh, strong_thresh)`
- `set_angle_range(start, end, step)`
- `set_scale_range(start, end, step)`
- `set_icp(enabled)`
- `create_model(image, mask=None, invalid_mask=None, model_name="classID_0")`
- `write_classes(path)`
- `read_classes(path)`
- `save_model(path)`
- `match(image, threshold=90.0, class_ids=[], mask=None, start_angle=-360, end_angle=360)`

