import json
import sqlite3
import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager

print("1. Foundry Local baslatiliyor...")
config = Configuration(app_name="rag-local-assistant")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

print("2. Embedding modeli hazirlaniyor...")
print("   -> qwen3-embedding-0.6b modeli kontrol ediliyor/indiriliyor...")
embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
embed_model.download()
print("   -> Embedding modeli basariyla indirildi ve yuklendi!")
embed_model.load()
embed_client = embed_model.get_embedding_client()

print("\n3. Chat modeli hazirlaniyor (ISTE EN COK BEKLETEN KISIM BURASI)...")
print("   -> qwen2.5-1.5b (yaklasik 1.8 GB) internetten indiriliyor...")
print(
    "   -> NOT: Bu islem internet hizina gore 3-5 dakika surebilir. Su an arka"
    " planda dosya cekiliyor, KESINLIKLE DONMADI!"
)
chat_model = manager.catalog.get_model("qwen2.5-1.5b")
chat_model.download()
print("   -> Harika! Model indirme bitti, simdi bellege aliniyor...")
chat_model.load()
print("   -> Chat modeli basariyla hazirlandi!\n")

# DEĞİŞİKLİK 1: RAG için hayati ayarlar! Halüsinasyonu ve kelime uydurmayı engeller.
chat_client = chat_model.get_chat_client()
chat_client.settings.max_tokens = 300
chat_client.settings.temperature = 0.1  # Yaraticiligi kistik, sadece gercege odaklanacak
chat_client.settings.top_p = 0.8
chat_client.settings.frequency_penalty = 0.2  # Tekrarlayan kelimeleri engeller

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

print("5. Dokuman okunuyor, parcalaniyor ve kaydediliyor...")
with open("dokuman.txt", "r", encoding="utf-8") as f:
  icerik = f.read()

dokumanlar = [parca.strip() for parca in icerik.split("\n\n") if parca.strip()]
print(f"   Dokuman {len(dokumanlar)} parcaya (chunk) bolundu.")

imlec.execute("DELETE FROM parcalar")
baglanti.commit()

for dokuman in dokumanlar:
  response = embed_client.generate_embedding(dokuman)
  vektor = response.data[0].embedding
  imlec.execute(
      "INSERT INTO parcalar (metin, embedding) VALUES (?, ?)",
      (dokuman, json.dumps(vektor)),
  )
baglanti.commit()
print(f"   {len(dokumanlar)} parca kaydedildi.")


def kosinus_benzerligi(v1, v2):
  v1 = np.array(v1)
  v2 = np.array(v2)
  return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# DEĞİŞİKLİK 2: Sadece 1 değil, en alakalı en iyi 2 parçayı (Top-2) getiriyoruz
def en_benzer_dokumanlari_bul(soru, top_k=2):
  soru_response = embed_client.generate_embedding(soru)
  soru_vektor = soru_response.data[0].embedding

  imlec.execute("SELECT metin, embedding FROM parcalar")
  kayitlar = imlec.fetchall()

  skorlu_kayitlar = []
  for metin, embedding_json in kayitlar:
    vektor = json.loads(embedding_json)
    skor = kosinus_benzerligi(soru_vektor, vektor)
    skorlu_kayitlar.append((metin, skor))

  skorlu_kayitlar.sort(key=lambda x: x[1], reverse=True)
  return skorlu_kayitlar[:top_k]


# DEĞİŞİKLİK 3: Modeli sınırlandıran, net ve kesin talimat yapısı
sistem_mesaji = (
    "Sen sadece sana verilen metinlere dayanarak cevap veren yardımcı bir"
    " asistansın.\n"
    "KURALLAR:\n"
    "1) Sadece aşağıdaki 'BAĞLAM' bölümünde verilen bilgileri kullan.\n"
    "2) Bağlamdaki bilgileri kendi cümlelerinle, düzgün ve akıcı bir Türkçe"
    " ile özetle.\n"
    "3) Bağlamda olmayan hiçbir bilgiyi, yorumu veya kendi bilgini ekleme.\n"
    "4) Asla kelimeleri yarım bırakma veya aynı kelimeleri arka arkaya tekrar"
    " etme."
)

ESIK_DEGERI = 0.35

print("\n6. Sistem hazir! Sorularinizi yazabilirsiniz.")
print("   Cikmak icin 'cikis' yazip Enter'a basin.\n")

while True:
  soru = input("Soru: ")

  if soru.strip().lower() == "cikis":
    print("Gorusmek uzere!")
    break

  en_iyi_sonuclar = en_benzer_dokumanlari_bul(soru, top_k=2)
  en_yuksek_skor = en_iyi_sonuclar[0][1] if en_iyi_sonuclar else 0

  print(f"   (En benzer dokuman skoru: {en_yuksek_skor:.4f})")

  if en_yuksek_skor < ESIK_DEGERI:
    print(
        "Cevap: Bu konuda elimde yeterli bilgi yok, bu yuzden cevap"
        " veremiyorum.\n"
    )
  else:
    birlestirilmis_baglam = "\n---\n".join(
        [item[0] for item in en_iyi_sonuclar]
    )

    messages = [
        {"role": "system", "content": sistem_mesaji},
        {
            "role": "user",
            "content": f"BAĞLAM:\n{birlestirilmis_baglam}\n\nSORU: {soru}",
        },
    ]
    response = chat_client.complete_chat(messages)
    ham_cevap = response.choices[0].message.content.strip()

    if "</think>" in ham_cevap:
      cevap = ham_cevap.split("</think>")[-1].strip()
    else:
      cevap = ham_cevap

    if cevap == "":
      cevap = (
          "(Ozet uretilemedi, ancak ilgili bilgi bulundu:"
          f" {en_iyi_sonuclar[0][0]})"
      )

    print(f"Cevap: {cevap}\n")

baglanti.close()