from foundry_local_sdk import Configuration, FoundryLocalManager

print("1. Ayarlar olusturuluyor...")
config = Configuration(app_name="rag-local-assistant")

print("2. Foundry Local baslatiliyor...")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

print("3. Embedding modeli seciliyor...")
model = manager.catalog.get_model("qwen3-embedding-0.6b")

print("4. Model indiriliyor (zaten indiyse hizli gecer)...")
model.download()

print("5. Model bellege yukleniyor...")
model.load()

print("6. Embedding istemcisi olusturuluyor...")
client = model.get_embedding_client()

print("7. Ornek cumleler icin embedding uretiliyor...")
cumleler = [
    "Kedi bir hayvandir.",
    "Kopek bir hayvandir.",
    "Bugun hava cok soguk."
]

for cumle in cumleler:
    response = client.generate_embeddings([cumle])
    vektor = response.data[0].embedding
    print(f"'{cumle}' -> vektorun ilk 5 sayisi: {vektor[:5]}")
    print(f"   Vektorun toplam uzunlugu: {len(vektor)}")


    import numpy as np

print("8. Benzerlik hesaplaniyor...")

def kosinus_benzerligi(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Her cumle icin vektorleri tekrar hesaplayip saklayalim
vektorler = []
for cumle in cumleler:
    response = client.generate_embeddings([cumle])
    vektorler.append(response.data[0].embedding)

print(f"\n'{cumleler[0]}' ile '{cumleler[1]}' benzerligi: {kosinus_benzerligi(vektorler[0], vektorler[1]):.4f}")
print(f"'{cumleler[0]}' ile '{cumleler[2]}' benzerligi: {kosinus_benzerligi(vektorler[0], vektorler[2]):.4f}")