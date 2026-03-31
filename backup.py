import os
import shutil
import time
from pathlib import Path

def backup_nexus_data(dest_folder=None):
    """NexusCore checkpoint, hafıza ve tokenizer verilerini sıkıştırıp yedekler."""
    
    # Yedeklenecek kaynak klasörler
    source_dirs = ["checkpoints", "memory_store", "tokenizer_data"]
    base_dir = Path(os.getcwd())
    
    # Destinasyon klasörü belirlenmediyse masaüstüne veya projenin içine atalım
    if not dest_folder:
        user_home = str(Path.home())
        # Eğer Google Drive kuruluysa orayı bulmaya çalışabilirsin
        # Örnek path: f"{user_home}/Library/CloudStorage/GoogleDrive-senin.email@gmail.com/My Drive/NexusBackups"
        
        # Şimdilik proje içinde backups klasörü oluşturalım
        dest_folder = base_dir / "backups"
    
    dest_path = Path(dest_folder)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Yedek adı için zaman damgası
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_filename = f"nexus_backup_{timestamp}"
    backup_zip_path = dest_path / backup_filename
    
    print(f"📦 Yedekleme başlatılıyor: {backup_filename}.zip")
    
    # Önce geçici bir klasörde toplayalım
    tmp_backup_dir = base_dir / "tmp_backup_build"
    tmp_backup_dir.mkdir(exist_ok=True)
    
    try:
        has_data = False
        for folder in source_dirs:
            src = base_dir / folder
            if src.exists():
                print(f"  -> Kopyalanıyor: {folder}/")
                shutil.copytree(src, tmp_backup_dir / folder)
                has_data = True
            else:
                print(f"  -> Bulunamadı (Atkandı): {folder}/")
        
        if not has_data:
            print("❌ Ziplenecek kaynak klasör bulunamadı!")
            return
            
        # Ziple
        print(f"🗜️ Sıkıştırılıyor...")
        shutil.make_archive(str(backup_zip_path), 'zip', tmp_backup_dir)
        
        print(f"✅ Yedekleme başarılı!")
        print(f"💾 Dosya: {backup_zip_path}.zip")
        print(f"☁️ Şimdi bu dosyayı Google Drive'a kopyalayabilirsiniz.")
        
    finally:
        # Geçici klasörü temizle
        if tmp_backup_dir.exists():
            shutil.rmtree(tmp_backup_dir)

if __name__ == "__main__":
    # Eğer GDrive yolun belliyse buraya yazabilirsin.
    # Örnek: gdrive_path = "/Users/greathill/Library/CloudStorage/GoogleDrive-seninhesap@gmail.com/My Drive/NexusAI_Models"
    
    backup_nexus_data()
