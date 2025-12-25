"""
🧠 CIFAR-10 Görüntü Sınıflandırma - Web Arayüzü
Görüntü sınıflandırma için Streamlit uygulaması
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# Sayfa Yapılandırması
st.set_page_config(
    page_title="YAPAY ZEKA DESTEKLİ GÖRÜNTÜ SINIFLANDIRMA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Tasarım için CSS
st.markdown("""
<style>
    /* Ana arka plan gradyanı */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Ana konteyner stillendirmesi */
    .main-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Başlık stillendirmesi */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .subtitle {
        color: #a0a0a0;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    
    /* Yükleme alanı stillendirmesi */
    .upload-container {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Sonuç kartı stillendirmesi */
    .result-card {
        background: linear-gradient(145deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .prediction-text {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fff;
        margin: 10px 0;
    }
    
    .confidence-text {
        font-size: 1.5rem;
        color: #667eea;
    }
    
    /* Buton stillendirmesi */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 50px;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 30px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Sınıf rozetleri */
    .class-badge {
        display: inline-block;
        padding: 8px 20px;
        margin: 5px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Bilgi kutuları */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 15px 20px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
    }
    
    /* Streamlit markalamasını gizle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Resim konteyneri */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Türkçe çeviriler ve emojilerle CIFAR-10 Sınıfları
CLASSES = {
    'airplane': {'tr': 'Uçak', 'emoji': '✈️'},
    'automobile': {'tr': 'Otomobil', 'emoji': '🚗'},
    'bird': {'tr': 'Kuş', 'emoji': '🐦'},
    'cat': {'tr': 'Kedi', 'emoji': '🐱'},
    'deer': {'tr': 'Geyik', 'emoji': '🦌'},
    'dog': {'tr': 'Köpek', 'emoji': '🐕'},
    'frog': {'tr': 'Kurbağa', 'emoji': '🐸'},
    'horse': {'tr': 'At', 'emoji': '🐴'},
    'ship': {'tr': 'Gemi', 'emoji': '🚢'},
    'truck': {'tr': 'Kamyon', 'emoji': '🚛'}
}

CLASS_NAMES = list(CLASSES.keys())
CLASS_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

@st.cache_resource
def load_model():
    """Eğitilmiş modeli önbelleğe alarak yükle"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

def preprocess_image(image):
    """Model tahmini için görüntüyü ön işle"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(model, image_tensor):
    """Tahmin yap ve olasılıkları döndür"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
    return predicted_idx.item(), confidence.item(), probabilities.numpy()

def create_probability_chart(probabilities):
    """Sınıf olasılıkları için etkileşimli çubuk grafik oluştur"""
    fig = go.Figure()
    
    sorted_indices = probabilities.argsort()[::-1]
    sorted_probs = probabilities[sorted_indices] * 100
    sorted_classes = [f"{CLASSES[CLASS_NAMES[i]]['emoji']} {CLASSES[CLASS_NAMES[i]]['tr']}" 
                      for i in sorted_indices]
    sorted_colors = [CLASS_COLORS[i] for i in sorted_indices]
    
    fig.add_trace(go.Bar(
        x=sorted_probs,
        y=sorted_classes,
        orientation='h',
        marker=dict(
            color=sorted_colors,
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        text=[f'{p:.1f}%' for p in sorted_probs],
        textposition='outside',
        textfont=dict(color='white', size=12)
    ))
    
    fig.update_layout(
        title=dict(
            text='📊 Sınıf Olasılıkları',
            font=dict(size=20, color='white'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Olasılık (%)', font=dict(color='white')),
            tickfont=dict(color='white'),
            gridcolor='rgba(255,255,255,0.1)',
            range=[0, max(sorted_probs) * 1.2]
        ),
        yaxis=dict(
            tickfont=dict(color='white', size=14),
            autorange='reversed'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=120, r=60, t=60, b=40),
        showlegend=False
    )
    
    return fig

# Ana Uygulama
def main():
    # Başlık
    st.markdown('<h1 class="main-title">🧠 YAPAY ZEKA DESTEKLİ GÖRÜNTÜ SINIFLANDIRMA</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Yapay Zeka ile Görsel Tanıma Sistemi</p>', unsafe_allow_html=True)
    
    # Mevcut sınıfları göster
    st.markdown("---")
    cols = st.columns(10)
    for i, (class_name, info) in enumerate(CLASSES.items()):
        with cols[i]:
            st.markdown(f"<div style='text-align: center; padding: 10px;'>"
                       f"<span style='font-size: 2rem;'>{info['emoji']}</span><br>"
                       f"<span style='color: white; font-size: 0.8rem;'>{info['tr']}</span>"
                       f"</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Modeli yükle
    model = load_model()
    
    # İki sütunlu düzen
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Görsel Yükle")
        st.markdown("<div class='info-box'>Bilgisayarınızdan bir görsel seçin (JPG, PNG, JPEG)</div>", 
                   unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Görsel seçmek için tıklayın veya sürükleyip bırakın",
            type=["jpg", "png", "jpeg"],
            help="Desteklenen formatlar: JPG, PNG, JPEG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📷 Yüklenen Görsel", use_container_width=True)
            
            # Tahmin butonu
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🔮 Tahmin Et", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Sonuçlar")
        
        if uploaded_file is not None and 'predict_btn' in dir() and predict_btn:
            with st.spinner("🔄 Görsel analiz ediliyor..."):
                # Ön işlem ve tahmin
                img_tensor = preprocess_image(image)
                pred_idx, confidence, probabilities = predict(model, img_tensor)
                
                pred_class = CLASS_NAMES[pred_idx]
                pred_info = CLASSES[pred_class]
                
                # Sonucu göster
                st.markdown(f"""
                <div class="result-card">
                    <span style="font-size: 4rem;">{pred_info['emoji']}</span>
                    <div class="prediction-text">{pred_info['tr']}</div>
                    <div class="confidence-text">Güven: %{confidence*100:.1f}</div>
                    <div style="color: #888; margin-top: 10px;">({pred_class})</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Olasılık grafiği
                st.markdown("<br>", unsafe_allow_html=True)
                fig = create_probability_chart(probabilities)
                st.plotly_chart(fig, use_container_width=True)
                
                # En yüksek 3 tahmin
                st.markdown("### 🏆 En Yüksek 3 Tahmin")
                top3_indices = probabilities.argsort()[::-1][:3]
                
                top_cols = st.columns(3)
                medals = ["🥇", "🥈", "🥉"]
                for i, idx in enumerate(top3_indices):
                    with top_cols[i]:
                        class_info = CLASSES[CLASS_NAMES[idx]]
                        st.markdown(f"""
                        <div style="text-align: center; padding: 15px; 
                                    background: rgba(255,255,255,0.05); 
                                    border-radius: 10px;">
                            <div style="font-size: 1.5rem;">{medals[i]}</div>
                            <div style="font-size: 2rem;">{class_info['emoji']}</div>
                            <div style="color: white; font-weight: 600;">{class_info['tr']}</div>
                            <div style="color: #667eea;">%{probabilities[idx]*100:.1f}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 60px 20px; 
                        background: rgba(255,255,255,0.03); 
                        border-radius: 15px; border: 1px dashed rgba(255,255,255,0.2);">
                <span style="font-size: 4rem;">🖼️</span>
                <p style="color: #888; margin-top: 20px;">
                    Sonuçlar burada görüntülenecek.<br>
                    Lütfen sol taraftan bir görsel yükleyin.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Alt Bilgi (Footer)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🤖 ResNet-18 Transfer Learning ile Eğitilmiş Model</p>
        <p>📊 CIFAR-10 Veri Seti (10 Sınıf, 60.000 Görsel)</p>
        <p style="margin-top: 10px; font-size: 0.8rem;">
            Powered by PyTorch & Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
