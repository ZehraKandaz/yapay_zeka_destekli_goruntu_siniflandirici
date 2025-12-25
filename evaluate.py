"""
CIFAR-10 Model Değerlendirme Betiği
Doğruluk (accuracy), kesinlik (precision), duyarlılık (recall), F1-skoru hesaplar ve karmaşıklık matrisi oluşturur
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import json
import os

# CIFAR-10 Sınıfları
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_model(model_path="model.pth"):
    """Eğitilmiş modeli yükle"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def get_test_loader():
    """CIFAR-10 test veri yükleyicisini al"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    
    test_data = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    
    return DataLoader(test_data, batch_size=64, shuffle=False)

def evaluate_model(model, test_loader, device="cpu"):
    """Modeli değerlendir, tahminleri ve etiketleri döndür"""
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("[*] Model test seti üzerinde değerlendiriliyor...")
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            if (i + 1) % 50 == 0:
                print(f"  {(i + 1) * 64} / {len(test_loader.dataset)} görüntü işlendi")
    
    return np.array(all_preds), np.array(all_labels)

def calculate_metrics(y_true, y_pred):
    """Tüm değerlendirme metriklerini hesapla"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='macro'),
        "precision_weighted": precision_score(y_true, y_pred, average='weighted'),
        "recall_macro": recall_score(y_true, y_pred, average='macro'),
        "recall_weighted": recall_score(y_true, y_pred, average='weighted'),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
    }
    
    # Sınıf bazlı metrikler
    per_class_precision = precision_score(y_true, y_pred, average=None)
    per_class_recall = recall_score(y_true, y_pred, average=None)
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    
    metrics["per_class"] = {}
    for i, class_name in enumerate(CLASSES):
        metrics["per_class"][class_name] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i])
        }
    
    return metrics

def print_results(metrics, conf_matrix):
    """Değerlendirme sonuçlarını formatlı bir şekilde yazdır"""
    print("\n" + "="*60)
    print("MODEL DEĞERLENDİRME SONUÇLARI")
    print("="*60)
    
    print(f"\n[+] Genel Doğruluk: {metrics['accuracy']*100:.2f}%")
    print(f"\n[*] Makro Metrikler:")
    print(f"   Kesinlik (Precision): {metrics['precision_macro']*100:.2f}%")
    print(f"   Duyarlılık (Recall):    {metrics['recall_macro']*100:.2f}%")
    print(f"   F1-Skoru:  {metrics['f1_macro']*100:.2f}%")
    
    print(f"\n[*] Ağırlıklı Metrikler:")
    print(f"   Kesinlik (Precision): {metrics['precision_weighted']*100:.2f}%")
    print(f"   Duyarlılık (Recall):    {metrics['recall_weighted']*100:.2f}%")
    print(f"   F1-Skoru:  {metrics['f1_weighted']*100:.2f}%")
    
    print("\n[*] Sınıf Bazlı Performans:")
    print("-"*50)
    print(f"{'Sınıf':<12} {'Kesinlik':<12} {'Duyarlılık':<12} {'F1-Skoru':<12}")
    print("-"*50)
    
    for class_name, class_metrics in metrics["per_class"].items():
        print(f"{class_name:<12} {class_metrics['precision']*100:>8.2f}%   {class_metrics['recall']*100:>8.2f}%   {class_metrics['f1']*100:>8.2f}%")
    
    print("\n[*] Karmaşıklık Matrisi (Confusion Matrix):")
    print(conf_matrix)
    print("="*60)

def save_metrics(metrics, conf_matrix, output_path="metrics.json"):
    """Metrikleri JSON dosyasına kaydet"""
    output = {
        "overall": {
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_macro": metrics["recall_macro"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"]
        },
        "per_class": metrics["per_class"],
        "confusion_matrix": conf_matrix.tolist(),
        "classes": CLASSES
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n[+] Metrikler şuraya kaydedildi: {output_path}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Kullanılan cihaz: {device}")
    
    # Load model
    model = load_model("model.pth")
    print("[+] Model başarıyla yüklendi")
    
    # Get test data
    test_loader = get_test_loader()
    print(f"[+] Test verisi yüklendi: {len(test_loader.dataset)} görüntü")
    
    # Evaluate
    y_pred, y_true = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Print and save results
    print_results(metrics, conf_matrix)
    save_metrics(metrics, conf_matrix)

if __name__ == "__main__":
    main()
