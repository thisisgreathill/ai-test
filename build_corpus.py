"""
build_corpus.py — Gerçek Türkçe corpus oluşturucu.

Vikipedi API'den Türkçe makaleler çeker ve eğitim için temiz text dosyası üretir.
"""

import urllib.request
import urllib.parse
import json
import re
import ssl
import time
from pathlib import Path

# Mac Python SSL sertifika sorunu çözümü
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE


def fetch_wikipedia_articles(num_articles=200, lang="tr"):
    """Vikipedi'den rastgele makaleler çek."""
    articles = []
    batch = 50  # API bir seferde max 50 veriyor
    
    print(f"  📥 Vikipedi'den {num_articles} makale çekiliyor ({lang})...")
    
    fetched = 0
    while fetched < num_articles:
        try:
            # Rastgele makale başlıkları al
            url = (
                f"https://{lang}.wikipedia.org/w/api.php?"
                f"action=query&list=random&rnnamespace=0"
                f"&rnlimit={min(batch, num_articles - fetched)}"
                f"&format=json"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "NexusCorpusBuilder/1.0"})
            with urllib.request.urlopen(req, timeout=15, context=ssl_ctx) as resp:
                data = json.loads(resp.read().decode())
            
            titles = [item["title"] for item in data["query"]["random"]]
            
            # Her makale için içerik çek
            for title in titles:
                try:
                    content = fetch_article_text(title, lang)
                    if content and len(content) > 100:
                        articles.append(content)
                        fetched += 1
                        if fetched % 20 == 0:
                            print(f"    ✅ {fetched}/{num_articles} makale")
                except Exception:
                    continue
            
            time.sleep(0.5)  # API rate limit
            
        except Exception as e:
            print(f"    ⚠️ Hata: {e}, tekrar deneniyor...")
            time.sleep(2)
    
    return articles


def fetch_article_text(title, lang="tr"):
    """Tek bir Vikipedi makalesinin düz metnini çek."""
    encoded = urllib.parse.quote(title)
    url = (
        f"https://{lang}.wikipedia.org/w/api.php?"
        f"action=query&titles={encoded}"
        f"&prop=extracts&explaintext=1&exsectionformat=plain"
        f"&format=json"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "NexusCorpusBuilder/1.0"})
    with urllib.request.urlopen(req, timeout=15, context=ssl_ctx) as resp:
        data = json.loads(resp.read().decode())
    
    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        text = page.get("extract", "")
        if text:
            return clean_text(text)
    return ""


def clean_text(text):
    """Metni temizle — gereksiz boşluklar, referanslar vs."""
    # Çoklu boşlukları tek boşluğa indir
    text = re.sub(r'\s+', ' ', text)
    # Çok kısa satırları at
    lines = text.split('. ')
    lines = [l.strip() for l in lines if len(l.strip()) > 20]
    text = '. '.join(lines)
    # Unicode normalleştir
    text = text.strip()
    return text


def build_corpus(num_articles=200, output_path="corpus.txt"):
    """Ana corpus oluşturma fonksiyonu."""
    print(f"""
    ╔═══════════════════════════════════════════════════╗
    ║      NexusCore — Corpus Builder                  ║
    ║      "Gerçek dünya bilgisiyle besle"             ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    articles = fetch_wikipedia_articles(num_articles)
    
    # Dosyaya yaz
    output = Path(output_path)
    with open(output, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(article + "\n")
    
    total_chars = sum(len(a) for a in articles)
    total_words = sum(len(a.split()) for a in articles)
    
    print(f"""
  ═══ Corpus Hazır ═══
  📄 Makale sayısı:  {len(articles)}
  📝 Toplam karakter: {total_chars:,}
  📖 Toplam kelime:   {total_words:,}
  💾 Dosya:          {output_path}
  📊 Dosya boyutu:   {output.stat().st_size / 1024:.1f} KB
    """)
    
    return str(output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", type=int, default=200)
    parser.add_argument("--output", type=str, default="corpus.txt")
    args = parser.parse_args()
    
    build_corpus(args.articles, args.output)
