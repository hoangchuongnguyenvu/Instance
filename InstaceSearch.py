import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
from scipy.spatial.distance import cosine
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import os
import torch
from torchvision import models
from torchvision import transforms
from netvlad import NetVLAD
import h5py
import matplotlib.pyplot as plt

class ImageSearcher:
    def __init__(self, chunks_dirs=['InstanceSearch/database_chunks_part1', 'InstanceSearch/database_chunks_part2']):
        print("Đang khởi tạo mô hình...")
        self.encoder = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.encoder = torch.nn.Sequential(*list(self.encoder.features.children()))
        
        self.net_vlad = NetVLAD(num_clusters=64, dim=512)
        self.model = torch.nn.Sequential(self.encoder, self.net_vlad)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("Đang tải dữ liệu...")
        self.load_database_chunks(chunks_dirs)
        print(f"Sử dụng thiết bị: {self.device}")

    def load_database_chunks(self, chunks_dirs):
        """
        Load và ghép các chunks của database
        Args:
            chunks_dirs (list): Danh sách các đường dẫn đến thư mục chứa chunks
        """
        self.features = []
        self.image_paths = []
        
        # Duyệt qua từng thư mục
        for chunks_dir in chunks_dirs:
            # Lấy danh sách các chunk files trong thư mục
            chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith('.h5')]
            chunk_files.sort()
            
            print(f"\nĐang tải từ {chunks_dir}")
            print(f"Tìm thấy {len(chunk_files)} chunks")
            
            for chunk_file in chunk_files:
                chunk_num = chunk_file.split('.')[0].split('_')[1]
                print(f"Đang tải chunk {chunk_num}...")
                
                # Load features
                with h5py.File(os.path.join(chunks_dir, f'chunk_{chunk_num}.h5'), 'r') as f:
                    chunk_features = f['features'][:]
                    self.features.append(chunk_features)
                
                # Load paths
                with open(os.path.join(chunks_dir, f'chunk_{chunk_num}_paths.pkl'), 'rb') as f:
                    chunk_paths = pickle.load(f)
                    self.image_paths.extend(chunk_paths)
        
        # Ghép các features lại
        self.features = np.concatenate(self.features, axis=0)
        
        # Chuẩn hóa features
        print("Đang chuẩn hóa features...")
        self.normalized_features = self.features / (np.linalg.norm(self.features, axis=1, keepdims=True) + 1e-8)
        
        print(f"\nĐã tải tổng cộng {len(self.image_paths)} ảnh")
        print(f"Kích thước feature: {self.features.shape}")

    def search(self, query_path, top_k=5):
        """
        Tìm kiếm ảnh tương tự
        Args:
            query_path (str): Đường dẫn đến ảnh query
            top_k (int): Số lượng kết quả trả về
        Returns:
            list: Danh sách các kết quả tìm kiếm
        """
        # Kiểm tra file tồn tại
        if not os.path.exists(query_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh: {query_path}")
        
        # Load và xử lý ảnh query
        query_img = Image.open(query_path).convert('RGB')
        query_tensor = self.transform(query_img)
        query_tensor = query_tensor.unsqueeze(0).to(self.device)
        
        # Trích xuất features
        with torch.no_grad():
            query_features = self.model(query_tensor).cpu().numpy()
        
        # Chuẩn hóa query features
        query_features = query_features.squeeze()
        query_features = query_features / (np.linalg.norm(query_features) + 1e-8)
        
        # Tính cosine similarity
        similarities = np.dot(self.normalized_features, query_features)
        
        # Chuyển sang khoảng cách
        distances = 1 - similarities
        
        # Thêm nhiễu nhỏ để tránh trường hợp bằng nhau
        distances += np.random.normal(0, 1e-6, distances.shape)
        
        # Lấy top-k kết quả
        indices = np.argsort(distances)[:top_k]
        results = []
        for idx in indices:
            results.append({
                'path': self.image_paths[idx],
                'distance': float(distances[idx])
            })
        
        return results

    def visualize_results(self, query_path, results):
        """
        Hiển thị ảnh query và các kết quả
        Args:
            query_path (str): Đường dẫn đến ảnh query
            results (list): Danh sách kết quả từ hàm search
        """
        # Đọc ảnh query
        query_img = cv2.imread(query_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # Tạo subplot
        plt.figure(figsize=(20, 4))
        
        # Hiển thị ảnh query
        plt.subplot(1, len(results) + 1, 1)
        plt.imshow(query_img)
        plt.title('Ảnh gốc')
        plt.axis('off')
        
        # Hiển thị kết quả
        for i, result in enumerate(results):
            try:
                img = cv2.imread(result['path'])
                if img is None:
                    raise Exception(f"Không thể tải ảnh: {result['path']}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                plt.subplot(1, len(results) + 1, i + 2)
                plt.imshow(img)
                plt.title(f'Kết quả {i+1}')
                plt.axis('off')
            except Exception as e:
                print(f"Lỗi hiển thị kết quả {i+1}: {str(e)}")
        
        plt.tight_layout()
        plt.show()

def main():
    st.set_page_config(page_title="Image Search", layout="wide")
    
    # Phần 1: Giới thiệu Dataset
    st.title("Demo Hệ thống Tìm kiếm Ảnh")
    st.header("1. Giới thiệu Dataset")
    st.write("""
    Dataset gồm 5000 ảnh đa dạng được thu thập từ COCO dataset, bao gồm:
    - Các đối tượng thường gặp trong cuộc sống hàng ngày
    - Phong cảnh thiên nhiên
    - Con người và động vật
    - Đồ vật, phương tiện giao thông
    - Các hoạt động và sự kiện
    
    Mỗi ảnh có kích thước và nội dung khác nhau, giúp đánh giá hiệu quả của hệ thống tìm kiếm trong nhiều tình huống khác nhau.
    """)
    
    # Danh sách tên các ảnh mẫu
    sample_images = [
        "InstanceSearch/DTS/000000001675.jpg", "InstanceSearch/DTS/000000001761.jpg", "InstanceSearch/DTS/000000001818.jpg", "InstanceSearch/DTS/000000001993.jpg", "InstanceSearch/DTS/000000002006.jpg",
        "InstanceSearch/DTS/000000002149.jpg", "InstanceSearch/DTS/000000002153.jpg", "InstanceSearch/DTS/000000002157.jpg", "InstanceSearch/DTS/000000002261.jpg", "InstanceSearch/DTS/000000002299.jpg"
    ]
    
    # Tạo 2 hàng, mỗi hàng 5 cột để hiển thị ảnh mẫu
    for row in range(2):
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            image_index = row * 5 + idx
            with col:
                try:
                    st.image(sample_images[image_index], 
                            caption=f"Ảnh mẫu {image_index + 1}",
                            use_column_width=True)
                except Exception as e:
                    st.error(f"Không thể load ảnh {sample_images[image_index]}")
    
    # Phần 2: Giới thiệu Quy trình
    st.header("2. Quy trình xử lý VLAD")
    
    # SIFT Feature Extraction
    st.subheader("2.1. Trích xuất đặc trưng SIFT (Scale Invariant Feature Transform)")
    st.write("""
    SIFT là phương pháp trích xuất đặc trưng bất biến với scale và rotation:
    
    1. *Scale-space Extrema Detection*
    - Tạo không gian scale bằng cách tích chập ảnh với Gaussian kernels
    - Tính DoG (Difference of Gaussian) giữa các scale liền kề
    - Tìm các điểm cực trị cục bộ trong không gian 3D (x, y, σ)
    
    2. *Keypoint Localization*
    - Loại bỏ các keypoint có độ tương phản thấp
    - Loại bỏ các điểm nằm dọc theo cạnh (sử dụng Hessian matrix)
    - Tinh chỉnh vị trí keypoint đến độ chính xác subpixel
    
    3. *Orientation Assignment*
    - Tính gradient magnitude và orientation cho mỗi pixel
    - Tạo histogram của các orientation (36 bins, mỗi bin 10 độ)
    - Gán orientation chính là peak của histogram
    
    4. *Keypoint Descriptor*
    - Chia vùng 16x16 xung quanh keypoint thành 16 ô 4x4
    - Tính 8-bin orientation histogram cho mỗi ô
    - Tạo vector 128 chiều (4x4x8) cho mỗi keypoint
    """)
    st.image("InstanceSearch/SIFT-feature-extraction-algorithm-process.png", 
             caption="Quy trình trích xuất đặc trưng SIFT", 
             use_column_width=True)
    
    # Visual Vocabulary Construction
    st.subheader("2.2. Vector of Locally Aggregated Descriptors (VLAD)")
    st.write("""
    VLAD là phương pháp mã hóa các SIFT descriptors thành một vector toàn cục:
    
    1. *Xây dựng Visual Dictionary*
    - Thu thập SIFT descriptors từ tập ảnh training
    - Áp dụng K-means clustering để tạo K centroids
    - Mỗi centroid đại diện cho một pattern đặc trưng cục bộ
    - K thường được chọn từ 32 đến 256
    
    2. *Quá trình mã hóa VLAD*
    - Với mỗi SIFT descriptor x trong ảnh:
        * Tìm centroid gần nhất c_i
        * Tính vector khác biệt v = x - c_i
        * Tích lũy v vào cluster tương ứng
    - Kết quả là K vectors, mỗi vector 128 chiều (kích thước SIFT)
    
    3. *Chuẩn hóa VLAD vector*
    - *Power normalization*: 
        * Áp dụng f(z) = sign(z)|z|^0.5
        * Giảm thiểu hiệu ứng bursty (nhiều descriptor giống nhau)
        * Làm mịn phân phối của các giá trị
    
    - *L2-normalization*:
        * Chuẩn hóa độ dài vector VLAD
        * Đảm bảo so sánh công bằng giữa các ảnh
        * Tối ưu cho việc tính cosine similarity
    """)
    st.image("InstanceSearch/OIP.jpg", 
             caption="Mã hóa VLAD", 
             use_column_width=True)
    
    st.subheader("2.3. Ưu điểm của SIFT-VLAD")
    st.write("""
    1. *Bất biến với các biến đổi hình học*
    - Scale invariant: nhờ đặc trưng SIFT
    - Rotation invariant: nhờ orientation assignment
    - Robust với thay đổi góc nhìn và ánh sáng
    
    2. *Biểu diễn đặc trưng phong phú*
    - Nắm bắt được cấu trúc cục bộ của ảnh qua SIFT
    - Tích lũy sự khác biệt first-order trong VLAD
    - Tạo ra vector đặc trưng phân biệt cao
    
    3. *Hiệu quả trong tìm kiếm*
    - Vector VLAD nhỏ gọn (K×128 chiều)
    - Tìm kiếm nhanh với cosine similarity
    - Phù hợp cho ứng dụng large-scale
    """)
    
    
    # Phần 3: Instance Search
    st.header("3. Instance Search")
    
    st.sidebar.title("Tùy chọn tìm kiếm")
    top_k = st.sidebar.slider("Số lượng kết quả", min_value=1, max_value=20, value=5)
    
    uploaded_file = st.file_uploader("Chọn ảnh để tìm kiếm...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Ảnh Query")
            query_image = Image.open(uploaded_file)
            st.image(query_image, use_column_width=True)
        
        try:
            # Lưu ảnh tạm thời
            temp_path = "temp_query.jpg"
            query_image.save(temp_path)
            
            with st.spinner('Đang tìm kiếm...'):
                searcher = ImageSearcher(['database_chunks_part1', 'database_chunks_part2'])
                results = searcher.search(temp_path, top_k=top_k)
            
            with col2:
                st.subheader("Kết quả tìm kiếm")
                if results:
                    cols = st.columns(3)
                    for idx, result in enumerate(results):
                        col_idx = idx % 3
                        with cols[col_idx]:
                            try:
                                img = Image.open(result['path'])
                                st.image(img,
                                       caption=f"Top {idx + 1}",
                                       use_column_width=True)
                            except Exception as e:
                                st.error(f"Không thể load ảnh: {result['path']}")
                else:
                    st.warning("Không tìm thấy ảnh tương tự!")
            
            # Xóa file tạm
            os.remove(temp_path)
                    
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {str(e)}")

if __name__ == "__main__":
    main()