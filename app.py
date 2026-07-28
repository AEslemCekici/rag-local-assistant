import json
import sqlite3
import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager
import streamlit as st

# Sayfa ayarları ve başlığı
st.set_page_config(page_title="Yerel RAG Asistanı", page_icon="🤖", layout="centered")

st.title("🤖 Offline RAG Soru-Cevap Asistanı")
st.caption(
    "Microsoft Foundry Local ve Qwen Modelleri ile İnternetsiz Çalışan Bilgi"
    " Asistanı"
)


# Modelleri ve Veritabanını Sadece 1 Kez Yüklemek için (Streamlit Cache)
@st.cache_resource
def sistemi_baslat():
  with st.spinner(
      "Foundry Local ve Qwen modelleri arka planda yükleniyor... (Lütfen"
      " bekleyin)"
  ):
    config = Configuration(app_name="rag-local-assistant")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # Embedding modeli
    embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embed_model.download()
    embed_model.load()
    embed_client = embed_model.get_embedding_client()

    # Chat modeli
    chat_model = manager.catalog.get_model("qwen2.5-1.5b")
    chat_model.download()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    # RAG ayarları
    chat_client.settings.max_tokens = 300
    chat_client.settings.temperature = 0.1
    chat_client.settings.top_p = 0.8
    chat_client.settings.frequency_penalty = 0.2

    return embed_client, chat_client


# Sistemi başlat
embed_client, chat_client = sistemi_baslat()


# Benzer dokümanları bulma fonksiyonu
def en_benzer_dokumanlari_bul(soru, top_k=2):
  soru_response = embed_client.generate_embedding(soru)
  soru_vektor = soru_response.data[0].embedding

  baglanti = sqlite3.connect("belgeler.db")
  imlec = baglanti.cursor()
  imlec.execute("SELECT metin, embedding FROM parcalar")
  kayitlar = imlec.fetchall()
  baglanti.close()

  skorlu_kayitlar = []
  for metin, embedding_json in kayitlar:
    vektor = json.loads(embedding_json)
    v1 = np.array(soru_vektor)
    v2 = np.array(vektor)
    skor = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    skorlu_kayitlar.append((metin, skor))

  skorlu_kayitlar.sort(key=lambda x: x[1], reverse=True)
  return skorlu_kayitlar[:top_k]


# Sohbet geçmişini ekranda tutmak için hafıza oluştur
if "mesajlar" not in st.session_state:
  st.session_state.mesajlar = [
      {
          "role": "assistant",
          "content": (
              "Merhaba! Veritabanındaki ders notları hakkında bana dilediğini"
              " sorabilirsin."
          ),
      }
  ]

# Eski sohbet mesajlarını ekrana çiz
for mesaj in st.session_state.mesajlar:
  with st.chat_message(mesaj["role"]):
    st.markdown(mesaj["content"])

# Kullanıcının alttaki kutuya soru yazmasını bekle
if soru := st.chat_input("Ders notları hakkında bir soru sorun..."):
  # Kullanıcının sorusunu ekrana yaz ve hafızaya ekle
  st.session_state.mesajlar.append({"role": "user", "content": soru})
  with st.chat_message("user"):
    st.markdown(soru)

  # Asistanın cevabı için alan aç
  with st.chat_message("assistant"):
    with st.spinner("Belgeler taranıyor ve cevap üretiliyor..."):
      en_iyi_sonuclar = en_benzer_dokumanlari_bul(soru, top_k=2)
      en_yuksek_skor = en_iyi_sonuclar[0][1] if en_iyi_sonuclar else 0

      if en_yuksek_skor < 0.35:
        cevap = (
            "Bu konuda veritabanımdaki belgelerde yeterli bilgi yok, bu"
            " yüzden cevap veremiyorum."
        )
      else:
        birlestirilmis_baglam = "\n---\n".join(
            [item[0] for item in en_iyi_sonuclar]
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
            " arkaya tekrar etme."
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
              f" {en_iyi_sonuclar[0][0]})"
          )

      st.markdown(cevap)

  # Asistanın cevabını da hafızaya ekle
  st.session_state.mesajlar.append({"role": "assistant", "content": cevap})