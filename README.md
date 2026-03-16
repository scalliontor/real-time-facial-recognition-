## Phần 1 — Ứng Dụng & Tính Năng

### Giới thiệu

Đây là hệ thống nhận diện khuôn mặt theo thời gian thực, được xây dựng với mục tiêu tạo ra **personalized experience** cho các artwork và installation art. Khi một người bước vào khung hình, hệ thống tức thời nhận ra họ là ai — hoặc tự động đăng ký nếu là người lạ — từ đó cho phép tác phẩm phản ứng và cá nhân hóa trải nghiệm theo từng người xem.

---

### 📸 Demo

*(Tính năng nhận diện sẽ hiển thị trực tiếp trên luồng camera, sử dụng các khung màu để phân biệt: đỏ (chưa đủ gần), cam (đang đăng ký), xanh lá (đã nhận diện))*

---

> ### Repo
> [*github.com/scalliontor/real-time-facial-recognition-*](https://github.com/scalliontor/real-time-facial-recognition-)

### Các tính năng chính

**1. Nhận diện khuôn mặt theo thời gian thực**
Hệ thống quét camera liên tục, phát hiện và nhận diện khuôn mặt ngay lập tức mà không có độ trễ đáng kể. Mỗi người được nhận ra sẽ hiển thị khung màu **xanh lá** kèm ID và độ chính xác.

**2. Tự động đăng ký người lạ (Auto-Registration)**
Khi phát hiện một khuôn mặt chưa có trong cơ sở dữ liệu:
- Nếu người đó đứng **quá xa** → khung **đỏ**, chờ họ đến gần hơn
- Khi đã đủ gần và **đứng yên khoảng 1 giây** → khung **vàng** với thanh tiến trình
- Hệ thống chụp và tổng hợp **5 frame** để tạo một "khuôn mặt đại diện" → khung **cam** hiển thị tiến độ (ví dụ: 3/5)
- Sau khi hoàn tất → khung **xanh lá** nhấp nháy với chữ "REGISTERED!"
Toàn bộ quá trình hoàn toàn tự động, không cần thao tác thủ công.

**3. Nhận diện nhiều người cùng lúc**
Hệ thống xử lý và theo dõi nhiều khuôn mặt độc lập trong cùng một khung hình. Mỗi người có trạng thái riêng biệt và không ảnh hưởng lẫn nhau.

**4. Quản lý danh sách người dùng**
Từ giao diện chính, có thể:
- Xem danh sách tất cả người đã đăng ký
- Xóa một người cụ thể khỏi hệ thống
- Xóa toàn bộ cơ sở dữ liệu để reset

---

### 🔗 Tích hợp vào TouchDesigner

Hệ thống này có thể đóng vai trò là **data source** cho TouchDesigner thông qua giao thức OSC (Open Sound Control) hoặc WebSocket. Workflow tích hợp cơ bản như sau:

**Bước 1 — Chạy song song**
Khởi động script Python và TouchDesigner đồng thời trên cùng một máy (hoặc hai máy trong cùng mạng LAN).

**Bước 2 — Output dữ liệu từ Python**
Thêm một module gửi OSC vào `recognize.py`. Mỗi khi nhận ra hoặc đăng ký một khuôn mặt, script gửi một message OSC chứa:
- `user_id` — UUID duy nhất của người đó
- `confidence` — độ chính xác nhận diện
- `status` — trạng thái: `recognized` / `registering` / `registered`
- `bbox` — tọa độ vùng khuôn mặt trên màn hình (tuỳ chọn)

**Bước 3 — Nhận dữ liệu trong TouchDesigner**
Dùng **OSC In CHOP** trong TouchDesigner để nhận các giá trị này. Từ đây, `user_id` trở thành trigger để:
- Gọi đúng bộ visual/artwork được cá nhân hóa cho người đó
- Điều khiển parameter, texture, animation theo profile người xem
- Lưu lại lịch sử tương tác theo từng UUID

**Bước 4 — Mapping theo người dùng**
Trong TouchDesigner, xây dựng một lookup table đơn giản: mỗi `user_id` map sang một bộ preset (màu sắc, âm thanh, nội dung artwork). Khi cùng người quay lại, họ nhận được đúng trải nghiệm của mình.

> 💡 **Gợi ý:** Nếu muốn tích hợp sâu hơn, có thể dùng `TouchDesigner + Python Script DAT` để gọi trực tiếp API của hệ thống, hoặc dùng shared memory / named pipe nếu cần latency cực thấp.


---

## Phần 2 — Hiệu Năng & Benchmark

### Thông số test machine

| Thành phần | Thông tin |
| --- | --- |
| Máy chủ | Local |
| GPU | NVIDIA GeForce RTX 3090 Ti |
| Driver Version | 555.99 |
| OS | Windows |
| Python version | 3.12+ |
| CUDA version | 12.5 |
| cuDNN version | 9.x |

---

### Kết quả benchmark

### FPS theo số người trong frame

*Dữ liệu đo được từ test thực tế 300 frame bằng `benchmark.py` trên GPU RTX 3090 Ti.*

| Số khuôn mặt trong frame | FPS trung bình | FPS min | FPS max |
| --- | --- | --- | --- |
| 1 người | **27.9** | 12.8 | 32.8 |
| 2 người | **25.7** | 0.8 | 43.3 |
| 3 người | **22.9** | 6.4 | 45.8 |
| 4 người | **23.8** | 20.1 | 31.8 |

### Độ trễ AI Engine (Latency)

| Metric | Giá trị đo được | Ghi chú |
| --- | --- | --- |
| Thời gian phát hiện (RetinaFace Detection) | **11.10** ms | Rất nhanh, chạy trên mọi frame |
| Trích xuất 512-D embedding (AuraFace) | **19.71** ms | Chỉ chạy 1 lần cho track ID mới |
| Thời gian so sánh Cosine Similarity | **0.01** ms | Gần như tức thời |
| **Tổng latency end-to-end (Inference)** | **30.82** ms | Tổng thời gian một model cycle hoàn chỉnh |

### Sử dụng tài nguyên (khi chạy ổn định)

| Tài nguyên | Mức sử dụng |
| --- | --- |
| GPU VRAM | **3.22 GB** |
| GPU Utilization | **31.4 %** |
| CPU | **10.1 %** |
| RAM Hệ thống | ~ 16.6 GB |

---

### Hướng dẫn đo benchmark

Hệ thống cung cấp sẵn script `benchmark.py` để tự động thu thập thông số real-time trên máy của bạn.

```bash
# Chạy script benchmark độc lập (Ghi đè file báo cáo vào data/report)
run_benchmark.bat
```

---

## Phần 3 — Kỹ Thuật & Architecture

### Stack công nghệ

| Thành phần | Công nghệ | Vai trò |
| --- | --- | --- |
| Ngôn ngữ | Python 3.9 – 3.12 | Core runtime |
| Video capture | OpenCV | Đọc và hiển thị webcam |
| Face detection | RetinaFace (InsightFace) | Phát hiện vùng khuôn mặt |
| Face recognition | **AuraFace-v1** | Trích xuất 512-D embedding |
| Tracking | SORT + Hungarian Algorithm (SciPy) | Theo dõi ID qua các frame |
| Inference runtime | ONNXRuntime (CUDAExecutionProvider) | Chạy model trên GPU |
| Database | SQLite | Lưu embeddings và metadata |

### Mô hình AI — AuraFace-v1

Đây là mô hình trung tâm của hệ thống. AuraFace-v1 là một **Convolutional Neural Network (CNN)** thuộc hệ sinh thái **InsightFace**, có các đặc điểm:

- **Output:** Vector embedding 512 chiều (512-D) đại diện cho đặc trưng khuôn mặt
- **So sánh:** Dùng **cosine similarity** — hai khuôn mặt càng giống nhau thì vector càng gần nhau (similarity gần 1.0)
- **Lưu trữ:** Mỗi người được lưu một "master embedding" — trung bình cộng của 5 frame — dưới dạng BLOB trong SQLite, đã được normalize
- **Hiệu quả:** Model chỉ chạy **một lần** khi lần đầu phát hiện người mới. Sau đó, SORT tracker đảm nhiệm việc theo dõi → tối ưu FPS tối đa

---

### Kiến trúc hệ thống

```
Camera Input (OpenCV)
        │
        ▼
┌─────────────────────┐
│   RetinaFace        │  ← Phát hiện vùng khuôn mặt trong frame
│   (Face Detector)   │
└─────────────────────┘
        │ Bounding boxes
        ▼
┌─────────────────────┐
│   SORT Tracker      │  ← Gán và duy trì Track ID qua các frame
│   (Hungarian Algo)  │     (không cần chạy model lại)
└─────────────────────┘
        │ Track ID + crop
        ▼
┌─────────────────────┐
│   AuraFace-v1       │  ← Chỉ chạy lần đăng ký / nhận diện lại mỗi N frame
│   (512-D Embedding) │
└─────────────────────┘
        │ Embedding vector
        ▼
┌─────────────────────┐
│   SQLite Database   │  ← So sánh cosine similarity với toàn bộ DB
│   (Cosine Match)    │
└─────────────────────┘
        │
        ▼
  Recognized / Unknown → Auto-Register Pipeline
```

---

### Cấu trúc file

| File | Vai trò |
| --- | --- |
| `main.py` | CLI entry point, menu hệ thống, khởi tạo webcam |
| `benchmark.py`| Công cụ đo lường hiệu năng FPS, Latency và thu thập ảnh mẫu |
| `face_engine.py` | Wrapper InsightFace: tải AuraFace-v1, detect, extract embedding, cosine similarity |
| `recognize.py` | Core loop: `FastFaceTracker` class, quản lý trạng thái tracking (steady / capturing / identified) |
| `register.py` | Logic đăng ký thủ công (đã gộp vào recognize.py ở chế độ unified) |
| `database.py` | SQLite wrapper: lưu/truy vấn UUID và 512-D embedding |
| `run.bat` | Script khởi động Windows (tự động link cuDNN và venv) |
| `run_benchmark.bat` | Script chạy chế độ benchmark Windows |

---

### Lưu trữ dữ liệu

```
data/
├── faces.db                  # SQLite database chứa UUID + embeddings
└── registered_faces/
    ├── <uuid-1>.jpg          # Ảnh crop khuôn mặt khi đăng ký
    ├── <uuid-2>.jpg
    └── ...
```

> Ảnh crop được lưu riêng để có thể **re-extract embedding** nếu sau này nâng cấp model — không cần đăng ký lại từ đầu.

---

## Phần 4 — Nguồn Tham Khảo & Tác Giả

Hệ thống được phát triển dựa trên các framework và thuật toán mã nguồn mở hàng đầu:

- **InsightFace (RetinaFace)**: Dùng cho mô-đun phát hiện khuôn mặt cực nhanh (Face Detection). [GitHub](https://github.com/deepinsight/insightface)
- **AuraFace-v1 (fal-ai)**: Mô hình trích xuất đặc trưng khuôn mặt (Face Recognition) tạo ra embedding 512 chiều có độ chính xác cao. [HuggingFace](https://huggingface.co/fal/AuraFace-v1)
- **SciPy (linear_sum_assignment)**: Trái tim của thuật toán theo dõi khối Hungarian (SORT Tracker) cho việc tracking nhiều ID ở mức FPS tối đa băng việc bỏ qua suy luận CNN. [SciPy](https://scipy.org/)
- **ONNX Runtime (GPU)**: Inference engine cốt lõi cho việc tăng tốc phần cứng bằng NVIDIA CUDA & cuDNN. [ONNX](https://onnxruntime.ai/)
- **OpenCV**: Xử lý đồ họa thời gian thực, webcam input và Giao diện người dùng. [OpenCV](https://opencv.org/)
