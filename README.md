# HDF5DataVisualize

HDF5 文件浏览与可视化服务（Flask）。在 EasyTeleop-AIO 中作为 `hdf5` 服务使用，默认对外端口为 `5000`。

## 运行方式

### Docker（推荐）

在 AIO 仓库根目录：

```bash
docker compose up -d hdf5
```

访问：

- Web 页面：`http://localhost:5000/`

### 本地运行（开发）

```bash
cd HDF5DataVisualize
python view_hdf5.py
```

## 数据目录

默认读取/写入目录为 `HDF5DataVisualize/databases/`（Compose 会挂载到容器内 `/app/databases`）。

## 主要接口（摘自 `view_hdf5.py`）

- `GET /`：页面入口
- `GET /api/database_folders`：列出 `databases/` 下子目录
- `GET /api/database_files/<folder_name>`：列出该目录下 `.hdf5` 文件
- `POST /api/process_hdf5`：解析指定 HDF5（按摄像头分组返回 base64 图片序列）
- 上传：
  - `POST /api/upload_chunk`
  - `POST /api/merge_chunks`（输出到 `databases/uploaded_from_api/`）

更多 AIO 集成说明见：`doc/modules/hdf5.md`

