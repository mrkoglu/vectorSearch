import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import gradio as gr

# Milvus Client'ı başlat
client = MilvusClient("milvus_demo.db")

# Eğer mevcutsa koleksiyonu sil
if client.has_collection(collection_name="image_embeddings"):
    client.drop_collection(collection_name="image_embeddings")

# Yeni koleksiyon oluştur
client.create_collection(
    collection_name="image_embeddings",
    dimension=768,  # CLIP Boyutu
)

# CLIP modelini yükle (en ağır model kullanıldı)
model = SentenceTransformer('clip-ViT-L-14')

# örüntüleri vektörlere dönüştürme
def generate_clip_embeddings(images_path, model):
    image_paths = [os.path.join(images_path, filename) for filename in os.listdir(images_path)]
    embeddings = []
    for img_path in image_paths:
        image = Image.open(img_path)
        embedding = model.encode(image)
        embeddings.append(embedding)
    return embeddings, image_paths

image_folder = '/home/mrk/Desktop/saat_case/download/download'
IMAGES_PATH = image_folder

# Embedding vektörlerini oluşturup ver tabanına ekle
embeddings, image_paths = generate_clip_embeddings(IMAGES_PATH, model)

data = [
    {"id": i, "vector": embeddings[i], "image_path": image_paths[i]}
    for i in range(len(image_paths))
]

# Verileri Milvus(veritabanı) koleksiyonuna ekleyin
client.insert(collection_name="image_embeddings", data=data)

# Milvus veritabanında vektörleri sorgulama
def search_vectors(query_vector, top_k=5):
    query_vector = np.array(query_vector).astype(np.float32).tolist()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    results = client.search(
        collection_name="image_embeddings",
        data=[query_vector],
        vector_field="vector",
        limit=top_k,
        params=search_params,
        output_fields=["image_path"]
    )

    return results

# Görüntü sorgulama fonksiyonu
def retrieve_similar_images(query, top_k=3):
    if isinstance(query, str):
        if query.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            # Görüntü dosya yolunu yükleyin
            try:
                query_image = Image.open(query)
                query_features = model.encode(query_image)
            except FileNotFoundError:
                return None, []
        else:
            # Metin tabanlı sorgu
            query_features = model.encode(query)
    else:
        return None, []

    results = search_vectors(query_features, top_k=top_k)

    if not results or not results[0]:
        return None, []

    results_data = results[0]
    retrieved_images = [item["entity"]["image_path"] for item in results_data]
    
    return query, retrieved_images


# arayüz (GRADIO)
def search_and_visualize(query, top_k):
    query, retrieved_images = retrieve_similar_images(query, top_k)
    if not retrieved_images:
        return "No images found", None

    imgs = [Image.open(img_path) for img_path in retrieved_images]
    return "Images found", imgs

gr.Interface(
    fn=search_and_visualize,
    inputs=[gr.Textbox(label="Text or Image Path"), gr.Slider(1, 10, value=3, label="Top K")],
    outputs=[gr.Textbox(label="Search Results"), gr.Gallery(label="Retrieved Images")]
).launch()
