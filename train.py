import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import sys

def train():
    # Ayarlar
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Cihaz: {device}", flush=True)

    # EĞİTİM için Gelişmiş Veri Artırma ile TRANSFORM
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Test için Transform (veri artırma yok)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # VERİ SETİ
    print("[*] Veri seti yükleniyor...", flush=True)
    train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    # Windows hızı için num_workers=0
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

    # MODEL (ResNet18 Fine-tuning)
    print("[*] Model hazırlanıyor (ResNet18 Fine-tuning)...", flush=True)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Tüm katmanları eğitime açıyoruz (Fine-tuning)
    for param in model.parameters():
        param.requires_grad = True

    # Son katmanı CIFAR-10'a göre güncelliyoruz
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    # Loss ve Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 2e-4}
    ], lr=1e-6)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # EĞİTİM DÖNGÜSÜ
    epochs = 10 # CPU üzerinde çok süreceği için epoch sayısını 10'a çektim
    print(f"[*] Eğitim başlıyor ({epochs} epoch)...", flush=True)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%", flush=True)
                
        train_acc = 100. * correct / total
        
        # Doğrulama Aşaması
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * val_correct / val_total
        
        print(f"==> Epoch {epoch+1} Bitti | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%", flush=True)
        
        if val_acc > best_acc:
            print(f"  [+] Yeni en iyi model bulundu (%{val_acc:.2f}), kaydediliyor...", flush=True)
            best_acc = val_acc
            torch.save(model.state_dict(), "model.pth")
            
        scheduler.step()

    print(f"\n[+] EĞİTİM TAMAMLANDI! En iyi Acc: %{best_acc:.2f}", flush=True)

if __name__ == '__main__':
    train()
