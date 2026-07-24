import sqlite3
import json
import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager

print("1. Foundry Local baslatiliyor...")
config = Configuration(app_name="rag-local-assistant")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

print("2. Embedding modeli hazirlaniyor...")
embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
embed_model.download()
embed_model.load()
embed_client = embed_model.get_embedding_client()

print("3. Chat modeli hazirlaniyor...")
chat_model = manager.catalog.get_model("qwen3-0.6b")
chat_model.download()
chat_model.load()
chat_client = chat_model.get_chat_client()

print("4. Veritabani hazirlaniyor...")
baglanti = sqlite3.connect("belgeler.db")
imlec = baglanti.cursor()
imlec.execute("""
    CREATE TABLE IF NOT EXISTS parcalar (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metin TEXT,
        embedding TEXT
    )
""")
baglanti.commit()

print("5. Ornek dokumanlar embedding'e cevriliyor ve kaydediliyor...")
print("   Dokuman dosyasi okunuyor...")
with open("dokuman.txt", "r", encoding="utf-8") as f:
    icerik = f.read()

dokumanlar = [parca.strip() for parca in icerik.split("\n\n") if parca.strip()]
print(f"   Dokuman {len(dokumanlar)} parcaya (chunk) bolundu.")

# Onceki denemelerden kalan kayitlari temizleyelim, tekrar tekrar eklenmesin
imlec.execute("DELETE FROM parcalar")
baglanti.commit()

for dokuman in dokumanlar:
    response = embed_client.generate_embedding(dokuman)
    vektor = response.data[0].embedding
    imlec.execute(
        "INSERT INTO parcalar (metin, embedding) VALUES (?, ?)",
        (dokuman, json.dumps(vektor))
    )
baglanti.commit()
print(f"   {len(dokumanlar)} dokuman kaydedildi.")

def kosinus_benzerligi(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def en_benzer_dokumani_bul(soru):
    soru_response = embed_client.generate_embedding(soru)
    soru_vektor = soru_response.data[0].embedding

    imlec.execute("SELECT metin, embedding FROM parcalar")
    kayitlar = imlec.fetchall()

    en_iyi_metin = None
    en_iyi_skor = -1
    for metin, embedding_json in kayitlar:
        vektor = json.loads(embedding_json)
        skor = kosinus_benzerligi(soru_vektor, vektor)
        if skor > en_iyi_skor:
            en_iyi_skor = skor
            en_iyi_metin = metin

    return en_iyi_metin, en_iyi_skor

print("\n6. Soru soruluyor...")
soru = "Malazgirt Savasi ne zaman oldu?"
bulunan_metin, skor = en_benzer_dokumani_bul(soru)
print(f"   Soru: {soru}")
print(f"   En benzer dokuman (skor {skor:.4f}): {bulunan_metin}")

print("\n7. Model, bulunan bilgiyle cevap uretiyor...")
sistem_mesaji = (
    "Sana verilen bilgiyi kullanarak soruyu cevapla. "
    "Eger bilgi soruyla ilgili degilse, bilmiyorum de."
)
messages = [
    {"role": "system", "content": sistem_mesaji},
    {"role": "user", "content": f"Bilgi: {bulunan_metin}\n\nSoru: {soru}"}
]
response = chat_client.complete_chat(messages)
print(f"\nModel cevabi: {response.choices[0].message.content}")

baglanti.close()