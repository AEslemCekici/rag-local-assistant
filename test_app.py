import unittest
from app import akilli_parcalama

class TestRAGSistemi(unittest.TestCase):
    
    def test_akilli_parcalama_uzunlugu(self):
        # Bilerek 100 kelimelik uzun ve sahte bir metin oluşturuyoruz
        ornek_metin = " ".join(["test"] * 100)
        
        # Fonksiyonumuzu çağırıp metni max 40 kelimelik parçalara bölmesini istiyoruz
        parcalar = akilli_parcalama(ornek_metin, max_kelime=40, kesisim=10)
        
        # 1. TEST: Fonksiyon boş liste mi döndürdü?
        self.assertTrue(len(parcalar) > 0, "HATA: Parçalama işlemi başarısız, liste boş!")
        
        # 2. TEST: Parçaların boyutu gerçekten istediğimiz sınırı (40) aşmamış mı?
        kelime_sayisi = len(parcalar[0].split())
        self.assertTrue(kelime_sayisi <= 40, f"HATA: Parça boyutu {kelime_sayisi} kelime oldu, sınırı aştı!")
        
        print("✅ Akıllı Parçalama (Chunking) testi başarıyla geçildi!")

if __name__ == '__main__':
    unittest.main()