from foundry_local_sdk import Configuration, FoundryLocalManager

print("1. Ayarlar olusturuluyor...")
config = Configuration(app_name="rag-local-assistant")

print("2. Foundry Local baslatiliyor...")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

print("3. Model seciliyor...")
model = manager.catalog.get_model("qwen3-0.6b")

print("4. Model indiriliyor (zaten indiyse hizli gecer)...")
model.download()

print("5. Model bellege yukleniyor...")
model.load()

print("6. Sohbet istemcisi olusturuluyor...")
client = model.get_chat_client()

print("7. Soru gonderiliyor...")
messages = [{"role": "user", "content": "Merhaba, calisiyor musun?"}]
response = client.complete_chat(messages)

print("Model cevabi:", response.choices[0].message.content)