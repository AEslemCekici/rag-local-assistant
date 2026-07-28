import json
import sqlite3
import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager
import streamlit as st

# 1. Sayfa Ayarları
st.set_page_config(page_title="Yerel RAG Asistanı", page_icon="🤖", layout="wide")

st.title("🤖 Offline RAG Soru-Cevap Asistanı")
st.caption(
    "Microsoft Foundry Local & Qwen Modelleri — Kendi Bilgi Bankanızı Yükleyin ve"
    " Sorgulayın!"
)


# 2. Modelleri Yükleme (Sadece 1 Kez Çalışır)
@st.cache_resource
def sistemi_baslat():
  with st.spinner("Foundry Local modelleri belleğe alınıyor... (Lütfen bekleyin)"):
    config = Configuration(app_name="rag-local-assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embed_model.download()
    embed_model.load()
    embed_client = embed_model.get_embedding_client()

    chat_model = manager.catalog.get_model("qwen2.5-1.5b")
    chat_model.download()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    chat_client.settings.max_tokens = 350
    chat_client.settings.temperature = 0.1
    chat_client.settings.top_p = 0.8
    chat_client.settings.frequency_penalty = 0.2

    return embed_client, chat_client


embed_client, chat_client = sistemi_baslat()


# 3. Veritabanına Yeni Doküman Ekleme Fonksiyonu
def dokuman_isleme_ve_kaydetme(metin_icerigi, dosya_adi):
  baglanti = sqlite3.connect("belgeler.db")
  imlec = baglanti.cursor()
  imlec.execute("""
        CREATE TABLE IF NOT EXISTS parcalar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dosya_adi TEXT,
            metin TEXT,
            embedding TEXT
        )
    """)

  # Eski tabloda dosya_adi sütunu yoksa hata vermemesi için basit kontrol
  try:
    imlec.execute("ALTER TABLE parcalar ADD COLUMN dosya_adi TEXT")
  except sqlite3.OperationalError:
    pass  # Sütun zaten varsa devam et

  # Metni paragraflara böl (Chunking)
  dokumanlar = [
      parca.strip() for parca in metin_icerigi.split("\n\n") if parca.strip()
  ]

  eklenen_sayisi = 0
  for dokuman in dokumanlar:
    # Aynı metin daha önce eklenmiş mi kontrol et (Çift kaydı engelle)
    imlec.execute("SELECT id FROM parcalar WHERE metin = ?", (dokuman,))
    if not imlec.fetchone():
      response = embed_client.generate_embedding(dokuman)
      vektor = response.data[0].embedding
      imlec.execute(
          "INSERT INTO parcalar (dosya_adi, metin, embedding) VALUES (?, ?, ?)",
          (dosya_adi, dokuman, json.dumps(vektor)),
      )
      eklenen_sayisi += 1

  baglanti.commit()
  baglanti.close()
  return eklenen_sayisi, len(dokumanlar)


# 4. Sol Menü (Sidebar) — Doküman Yükleme Alanı
with st.sidebar:
  st.header("📂 Bilgi Bankası Yönetimi")
  st.write(
      "Asistanın cevaplayabilmesi için buraya `.txt` formatında ders notu,"
      " makale veya özet yükleyin."
  )

  yuklenen_dosyalar = st.file_uploader(
      "Metin Dosyalarını Seçin",
      type=["txt"],
      accept_multiple_files=True,
  )

  if yuklenen_dosyalar:
    if st.button("🚀 Dosyaları Veritabanına İşle", use_container_width=True):
      toplam_eklenen = 0
      with st.spinner("Dosyalar okunuyor, parçalanıyor ve vektörleniyor..."):
        for dosya in yuklenen_dosyalar:
          dosya_icerigi = dosya.read().decode("utf-8")
          eklenen, toplam = dokuman_isleme_ve_kaydetme(
              dosya_icerigi, dosya.name
          )
          toplam_eklenen += eklenen
      st.success(f"✅ İşlem Tamam! {toplam_eklenen} yeni bilgi parçası kaydedildi.")

  st.divider()
  # Veritabanındaki mevcut kayıt sayısını göster
  baglanti = sqlite3.connect("belgeler.db")
  imlec = baglanti.cursor()
  try:
    imlec.execute("SELECT COUNT(*), COUNT(DISTINCT dosya_adi) FROM parcalar")
    toplam_parca, toplam_dosya = imlec.fetchone()
    st.metric("Veritabanındaki Belgeler", f"{toplam_dosya} Dosya")
    st.metric("Toplam Bilgi Parçası (Chunk)", f"{toplam_parca} Parça")
  except sqlite3.OperationalError:
    st.info("Henüz veritabanı oluşturulmadı.")
  baglanti.close()

  if st.button("🗑️ Sohbet Geçmişini Temizle", use_container_width=True):
    st.session_state.mesajlar = []
    st.rerun()


# 5. Benzer Dokümanları Bulma Fonksiyonu
def en_benzer_dokumanlari_bul(soru, top_k=2):
  soru_response = embed_client.generate_embedding(soru)
  soru_vektor = soru_response.data[0].embedding

  baglanti = sqlite3.connect("belgeler.db")
  imlec = baglanti.cursor()
  imlec.execute("SELECT dosya_adi, metin, embedding FROM parcalar")
  kayitlar = imlec.fetchall()
  baglanti.close()

  if not kayitlar:
    return []

  skorlu_kayitlar = []
  for dosya_adi, metin, embedding_json in kayitlar:
    vektor = json.loads(embedding_json)
    v1 = np.array(soru_vektor)
    v2 = np.array(vektor)
    skor = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    skorlu_kayitlar.append((dosya_adi, metin, skor))

  skorlu_kayitlar.sort(key=lambda x: x[2], reverse=True)
  return skorlu_kayitlar[:top_k]


# 6. Sohbet Ekranı Yönetimi
if "mesajlar" not in st.session_state:
  st.session_state.mesajlar = [
      {
          "role": "assistant",
          "content": (
              "Merhaba! Sol menüden yeni belgeler yükleyebilir veya"
              " veritabanındaki mevcut bilgiler hakkında bana sorular"
              " sorabilirsin."
          ),
      }
  ]

for mesaj in st.session_state.mesajlar:
  with st.chat_message(mesaj["role"]):
    st.markdown(mesaj["content"])

if soru := st.chat_input("Bilgi bankasında ne aramak istersiniz?"):
  st.session_state.mesajlar.append({"role": "user", "content": soru})
  with st.chat_message("user"):
    st.markdown(soru)

  with st.chat_message("assistant"):
    with st.spinner("Bilgiler taranıyor ve cevap hazırlanıyor..."):
      en_iyi_sonuclar = en_benzer_dokumanlari_bul(soru, top_k=2)
      en_yuksek_skor = en_iyi_sonuclar[0][2] if en_iyi_sonuclar else 0

      # Benzerlik eşiğini 0.40'a çektik (Daha kararlı çalışır)
      if en_yuksek_skor < 0.40:
        cevap = (
            "Bu konuda veritabanımdaki belgelerde yeterli bilgi bulamadım. Lütfen"
            " sol menüden bu konuyla ilgili bir makale veya ders notu yükleyip"
            " tekrar deneyin."
        )
      else:
        # Context hazırlama
        birlestirilmis_baglam = "\n---\n".join(
            [f"[Kaynak: {item[0]}] -> {item[1]}" for item in en_iyi_sonuclar]
        )

        sistem_mesaji = (
            "Sen sadece sana verilen metinlere dayanarak cevap veren yardımcı"
            " bir asistansın.\n"
            "KURALLAR:\n"
            "1) Sadece aşağıdaki 'BAĞLAM' bölümünde verilen bilgileri"
            " kullan.\n"
            "2) Bağlamdaki bilgileri kendi cümlelerinle, düzgün ve akıcı bir"
            " Türkçe ile özetle.\n"
            "3) Bağlamda olmayan hiçbir bilgiyi, yorumu veya kendi bilgini"
            " ekleme.\n"
            "4) Asla kelimeleri yarım bırakma veya aynı kelimeleri arka"
            " arkaya tekrar etme.\n"
            "5) Cevabının en altına bilgiyi aldığın dosyanın adını 'Kaynak:'"
            " şeklinde ekle."
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
              "(Özet üretilemedi, ancak ilgili bilgi bulundu:"
              f" {en_iyi_sonuclar[0][1]})"
          )

      st.markdown(cevap)

  st.session_state.mesajlar.append({"role": "assistant", "content": cevap})