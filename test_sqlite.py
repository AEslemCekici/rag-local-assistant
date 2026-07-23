import sqlite3
import json

print("1. Veritabanina baglaniliyor (yoksa yeni olusturuluyor)...")
baglanti = sqlite3.connect("belgeler.db")
imlec = baglanti.cursor()

print("2. Tablo olusturuluyor (yoksa)...")
imlec.execute("""
    CREATE TABLE IF NOT EXISTS parcalar (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metin TEXT,
        embedding TEXT
    )
""")

print("3. Ornek bir kayit ekleniyor...")
ornek_vektor = [0.023, 0.021, -0.003]
imlec.execute(
    "INSERT INTO parcalar (metin, embedding) VALUES (?, ?)",
    ("Kedi bir hayvandir.", json.dumps(ornek_vektor))
)
baglanti.commit()

print("4. Kayitlar okunuyor...")
imlec.execute("SELECT id, metin, embedding FROM parcalar")
sonuclar = imlec.fetchall()
for satir in sonuclar:
    id, metin, embedding_json = satir
    embedding = json.loads(embedding_json)
    print(f"ID: {id}, Metin: '{metin}', Embedding (ilk 3): {embedding[:3]}")

baglanti.close()
print("5. Baglanti kapatildi.")