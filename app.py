import json
import sqlite3
import numpy as np
import time
import logging
from foundry_local_sdk import Configuration, FoundryLocalManager
import streamlit as st
import PyPDF2
import docx
import os
from dotenv import load_dotenv

# Gizli ayarlari yukle
load_dotenv()

# ==============================================================================
# LOGLAMA (KARA KUTU) AYARLARI
# ==============================================================================
logging.basicConfig(
    filename="sistem.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)
logging.info("Sistem baslatiliyor...")

# ==============================================================================
# 1. MİNİMAL VE NATIVE ARAYÜZ
# ==============================================================================
st.set_page_config(page_title="Yerel RAG Asistanı", page_icon="⚡", layout="wide")

# ==============================================================================
# 2. MOTOR BAŞLATMA
# ==============================================================================
@st.cache_resource(show_spinner=False)
def motoru_baslat():
    config = Configuration(app_name="enterprise-local-rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embed_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embed_model.download(); embed_model.load()
    embed_client = embed_model.get_embedding_client()

    chat_model = manager.catalog.get_model("qwen2.5-1.5b")
    chat_model.download(); chat_model.load()
    chat_client = chat_model.get_chat_client()

    logging.info("Yapay zeka modelleri (Embedding ve Chat) basariyla yuklendi.")
    return embed_client, chat_client

embed_client, chat_client = motoru_baslat()

# ==============================================================================
# 3. VERİTABANI VE AKILLI PARÇALAMA
# ==============================================================================
DB_YOLU = os.getenv("DB_NAME", "kurumsal_belgeler.db")

def veritabani_hazirla():
    baglanti = sqlite3.connect(DB_YOLU)
    baglanti.execute("CREATE TABLE IF NOT EXISTS parcalar (id INTEGER PRIMARY KEY AUTOINCREMENT, dosya_adi TEXT, metin TEXT, embedding TEXT)")
    baglanti.commit(); baglanti.close()
    logging.info(f"Veritabani hazir: {DB_YOLU}")

veritabani_hazirla()

def dosya_icerigini_cikar(dosya):
    uzanti = dosya.name.split('.')[-1].lower()
    metin = ""
    try:
        if uzanti == "txt": metin = dosya.read().decode("utf-8")
        elif uzanti == "pdf":
            for sayfa in PyPDF2.PdfReader(dosya).pages: metin += sayfa.extract_text() + " "
        elif uzanti == "docx":
            metin = " ".join([para.text for para in docx.Document(dosya).paragraphs])
        logging.info(f"Dosya basariyla okundu: {dosya.name}")
    except Exception as e: 
        st.sidebar.error(f"Okuma hatası: {e}")
        logging.error(f"Dosya okuma hatasi ({dosya.name}): {e}")
    return metin

def akilli_parcalama(metin, max_kelime=150, kesisim=30):
    kelimeler = metin.split()
    parcalar = []
    for i in range(0, len(kelimeler), max_kelime - kesisim):
        parca = " ".join(kelimeler[i:i + max_kelime])
        if len(parca.strip()) > 20: 
            parcalar.append(parca)
    return parcalar

def dokuman_isle_ve_kaydet(metin_icerigi, dosya_adi):
    baglanti = sqlite3.connect(DB_YOLU)
    imlec = baglanti.cursor()
    dokumanlar = akilli_parcalama(metin_icerigi) 
    
    eklenen = 0
    for dokuman in dokumanlar:
        imlec.execute("SELECT id FROM parcalar WHERE metin = ? AND dosya_adi = ?", (dokuman, dosya_adi))
        if not imlec.fetchone():
            vektor = embed_client.generate_embedding(dokuman).data[0].embedding
            imlec.execute("INSERT INTO parcalar (dosya_adi, metin, embedding) VALUES (?, ?, ?)", (dosya_adi, dokuman, json.dumps(vektor)))
            eklenen += 1
    baglanti.commit(); baglanti.close()
    logging.info(f"Veritabanina islendi: {dosya_adi} | Eklenen chunk: {eklenen}")
    return eklenen

# ==============================================================================
# 4. SOL MENÜ: MÜHENDİSLİK KONTROL PANELİ
# ==============================================================================
with st.sidebar:
    st.title("⚙️ Kontrol Paneli")
    st.caption("Sistem durumunu ve model parametrelerini yönetin.")
    
    with st.container(border=True):
        st.markdown("**📡 Sistem Durumu**")
        st.success("✅ Foundry Motoru: Çevrimiçi")
        st.success("✅ Qwen2.5 LLM: Hazır")
        
    with st.container(border=True):
        st.markdown("**🧠 Model Parametreleri**")
        sicaklik = st.slider("Yaratıcılık (Temperature)", 0.0, 1.0, 0.3, 0.1)
        max_uzunluk = st.slider("Maksimum Yanıt Uzunluğu", 200, 1500, 800, 100)
        chat_client.settings.temperature = sicaklik
        chat_client.settings.max_tokens = max_uzunluk
        chat_client.settings.frequency_penalty = 1.2
        chat_client.settings.presence_penalty = 1.0

    with st.container(border=True):
        st.markdown("**📂 Bilgi Bankası Yönetimi**")
        yuklenen_dosyalar = st.file_uploader("Belge Ekle (PDF, DOCX, TXT)", type=["txt", "pdf", "docx"], accept_multiple_files=True, label_visibility="collapsed")
        
        if yuklenen_dosyalar and st.button("🚀 Sisteme İşle", type="primary", use_container_width=True):
            ilerleme_cubugu = st.progress(0, text="Belgeler işleniyor...")
            toplam = 0
            for i, dosya in enumerate(yuklenen_dosyalar):
                toplam += dokuman_isle_ve_kaydet(dosya_icerigini_cikar(dosya), dosya.name)
                ilerleme_cubugu.progress((i + 1) / len(yuklenen_dosyalar), text=f"{dosya.name} okundu.")
            st.success(f"✅ {toplam} yeni veri bloğu eklendi.")
            time.sleep(1)
            st.rerun()

        baglanti = sqlite3.connect(DB_YOLU)
        p_sayisi, d_sayisi = baglanti.execute("SELECT COUNT(*), COUNT(DISTINCT dosya_adi) FROM parcalar").fetchone()
        dosyalar = [row[0] for row in baglanti.execute("SELECT DISTINCT dosya_adi FROM parcalar").fetchall()]
        baglanti.close()

        st.metric(label="Veritabanı Doluluğu (Vektör Bloğu)", value=p_sayisi, delta=f"{d_sayisi} Kaynak")
        
        if dosyalar:
            with st.expander("📑 Yüklü Kaynakları İncele"):
                for d in dosyalar: st.markdown(f"- `{d}`")
                
            st.markdown("---")
            st.markdown("**🗑️ Veri Silme**")
            silinecek_dosya = st.selectbox("Silinecek dosyayı seçin:", dosyalar, label_visibility="collapsed")
            if st.button("Seçili Dosyayı Sil", use_container_width=True):
                baglanti = sqlite3.connect(DB_YOLU)
                baglanti.execute("DELETE FROM parcalar WHERE dosya_adi = ?", (silinecek_dosya,))
                baglanti.commit(); baglanti.close()
                logging.info(f"Kullanici dosyayi sildi: {silinecek_dosya}")
                st.success(f"{silinecek_dosya} başarıyla silindi!")
                time.sleep(1)
                st.rerun()
        else:
            st.info("Henüz belge yüklenmedi.")

    if st.button("💥 Tüm Veritabanını Sıfırla", type="primary", use_container_width=True):
        baglanti = sqlite3.connect(DB_YOLU)
        baglanti.execute("DELETE FROM parcalar")
        baglanti.commit(); baglanti.close()
        logging.warning("DIKKAT: Veritabani tamamen sifirlandi!")
        st.session_state.mesajlar = [{"role": "assistant", "content": "Veritabanı ve sohbet geçmişi tamamen temizlendi.", "kaynaklar": []}]
        st.rerun()

# ==============================================================================
# 5. ARAMA VE EFEKTLER
# ==============================================================================
def en_benzer_parcalari_bul(soru, top_k=3):
    soru_vektor = embed_client.generate_embedding(soru).data[0].embedding
    baglanti = sqlite3.connect(DB_YOLU)
    kayitlar = baglanti.execute("SELECT dosya_adi, metin, embedding FROM parcalar").fetchall()
    baglanti.close()
    if not kayitlar: return []

    skorlu = []
    for d_adi, metin, emb_json in kayitlar:
        v1, v2 = np.array(soru_vektor), np.array(json.loads(emb_json))
        skor = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        skorlu.append({"dosya": d_adi, "metin": metin, "skor": float(skor)})

    return sorted(skorlu, key=lambda x: x["skor"], reverse=True)[:top_k]

def daktilo_efekti(metin):
    for kelime in metin.split(" "): yield kelime + " "; time.sleep(0.015)

# ==============================================================================
# 6. ANA EKRAN (PROFESYONEL VE SADE)
# ==============================================================================
st.title("⚡ Yerel Bilgi Asistanı")
st.caption("Microsoft Foundry Local altyapısı ile çalışan, %100 çevrimdışı ve güvenli sorgulama sistemi.")
st.divider()

if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = [{"role": "assistant", "content": "Sistem hazır. Lütfen analiz edilecek belgeleri sol menüden yükleyin veya mevcut veritabanı üzerinden sorunuzu iletin.", "kaynaklar": []}]

for mesaj in st.session_state.mesajlar:
    with st.chat_message(mesaj["role"]):
        st.markdown(mesaj["content"])
        if mesaj.get("kaynaklar"):
            with st.expander("🔍 Kullanılan Kaynaklar"):
                for k in mesaj["kaynaklar"]: st.markdown(f"**{k['dosya']}** (%{k['skor']*100:.1f})\n*{k['metin']}*")

if soru := st.chat_input("Asistana sor..."):
    logging.info(f"Kullanici sorusu: {soru}")
    st.session_state.mesajlar.append({"role": "user", "content": soru, "kaynaklar": []})
    with st.chat_message("user"): st.markdown(soru)

    with st.chat_message("assistant"):
        bulunanlar = en_benzer_parcalari_bul(soru, top_k=3)
        
        # BARAJ BURADA 0.45'E ÇIKARILDI: Alakasız sorular doğrudan reddedilecek.
        if not bulunanlar or bulunanlar[0]["skor"] < 0.45:
            st.warning("Bu konuyla ilgili yüklenmiş belgelerde yeterli bilgi bulunamadı.")
            st.session_state.mesajlar.append({"role": "assistant", "content": "Bilgi bulunamadı.", "kaynaklar": []})
        else:
            baglam = "\n".join([f"[{p['dosya']}]: {p['metin']}" for p in bulunanlar])
            
            sistem_promptu = (
                "Sen kıdemli bir kurumsal bilgi asistanısın.\n"
                "KURALLAR:\n"
                "1) SADECE sana verilen BAĞLAM'daki verileri kullan.\n"
                "2) Eğer sorunun cevabı bağlamda açıkça yoksa 'Bilgi bulunamadı.' de.\n"
                "3) Bilgi verdiğinde mutlaka kaynağını belirt.\n"
                "4) Cevabını her zaman sade ve profesyonel bir formatta ver."
            )
            
            messages = [{"role": "system", "content": sistem_promptu}]
            gecmis = st.session_state.mesajlar[:-1] 
            for m in gecmis[-4:]: 
                messages.append({"role": m["role"], "content": m["content"]})
                
            messages.append({"role": "user", "content": f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"})

            ham_cevap = chat_client.complete_chat(messages).choices[0].message.content.strip()
            if "</think>" in ham_cevap: ham_cevap = ham_cevap.split("</think>")[-1].strip()

            st.write_stream(daktilo_efekti(ham_cevap))
            
            with st.expander("🔍 Kullanılan Kaynaklar"):
                for idx, k in enumerate(bulunanlar, 1): st.markdown(f"**[{idx}] {k['dosya']}** — Skoru: `% {k['skor']*100:.1f}`")
            
            st.session_state.mesajlar.append({"role": "assistant", "content": ham_cevap, "kaynaklar": bulunanlar})