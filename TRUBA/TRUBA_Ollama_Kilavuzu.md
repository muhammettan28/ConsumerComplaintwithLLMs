**TRUBA ARF Üzerinde Ollama ile GPU Inference Kılavuzu**  
Bu kılavuz, TRUBA ARF kümesinde Ollama kurup GPU üzerinde model  
   
 çalıştırmak (inference testi) içindir. Sıfırdan, adım adım ilerler.  
***Önce oku:*** * TRUBA'da iki tür sunucu var:*  
- ***Arayüz (arf-ui1 = 172.16.6.11):*** * dosya hazırlar, indirir, iş gönderirsin. * ***Ağır iş / inference ÇALIŞTIRILMAZ*** * (hesabın askıya alınabilir).*  
- ***Hesaplama node'ları (akya-cuda vb.):*** * asıl GPU işleri burada koşar. * ***Buralarda internet YOKTUR.***  
*Bu yüzden mantık şu: * ***modeli arayüzde indir → işi GPU kuyruğuna gönder → orada offline çalıştır.***  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OQQmAABRAsSdYxKY/jMFMIZ7ECt5E2BJsmZmt2gMA4C+Otbqr8+sJAACvXQ85QgYXd/O+eQAAAABJRU5ErkJggg==)  
**0. Başlamadan önce 3 şeyi doğrula**  
VPN açıkken arayüze bağlan:  
ssh muhtan@172.16.6.11  
   
Bağlandıktan sonra arayüzde şunları çalıştır:  
**a) Hesap adın (betikte ** **-A** ** için):**  
sacctmgr show assoc where user=$USER format=account,partition  
   
Çıktıdaki Account sütunu, betikte #SBATCH -A satırına yazacağın değerdir.  
   
 Genelde kullanıcı adınla aynıdır (muhtan), ama proje hesabıysa farklı olabilir.  
**b) GPU kuyruğuna erişimin var mı:**  
sinfo -p akya-cuda  
   
Satır dönerse erişimin var demektir.  
**c) Ev dizinin ve boş alanın:**  
echo $HOME  
 lssrv          # kullanım/kota durumun  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OYQ1AABSAwc8mi5wvkwZyCKCAACr4Z7a7BLfMzFYdAQDwF+da3dX+9QQAgNeuB6feBdUJcyS2AAAAAElFTkSuQmCC)  
**1. Hangi GPU kuyruğu? (Senin için tablo özeti)**  
| | | | | |  
|-|-|-|-|-|  
| **Kuyruk** | **GPU** | **VRAM** | **GPU başına çekirdek** | **Erişim** |   
| **akya-cuda** | 4× V100 | 16 GB | **10** | Herkese açık ✅ (önerilen) |   
| **barbun-cuda** | 2× P100 | 16 GB | **20** | Herkese açık |   
| palamut-cuda | 8× A100 | 80 GB | 16 | Sadece izinli projeler |   
| kolyoz-cuda | 4× H100/H200 | 80/141 GB | 16 | Sadece izinli projeler |   
   
**Başlangıç için: akya-cuda, 1 GPU.** 16 GB VRAM şu modelleri rahat çalıştırır:  
- 3B–8B (q4 kuantize): rahat  
- 13B–14B (q4): sığar, çalışır  
- 70B: **tek GPU'ya sığmaz** → ileride --gres=gpu:4 (4× V100 = 64 GB) ile denenebilir  
*GPU kuyruğunda iş * ***mutlaka*** * * *--gres=gpu:N* * ile GPU istemeli, yoksa reddedilir.*  
 *  
 Ayrıca GPU kuyruğunda * *-C weka* * veya * *--constraint=orfoz* * gibi satırlar * ***kullanma*** * (onlar CPU kuyrukları içindir).*  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AUBBAsUfyRTCh9VRgEBGsWGAjJK2CbjNzVGcAAPzFtapV7V9PAAB47X4AEWgEMAY9+pUAAAAASUVORK5CYII=)  
**2. Ollama'yı kur (arayüzde, internet olan yerde — bir kez)**  
TRUBA'da hazır Ollama modülü yok; statik ikiliyi ev dizinine kuracağız (root gerekmez).  
cd $HOME  
 mkdir -p ollama && cd ollama  
 curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama.tgz  
 tar -xzf ollama.tgz        # bin/ollama ve lib/ollama oluşur  
 rm ollama.tgz  
 export PATH=$HOME/ollama/bin:$PATH  
 ollama --version           # sürüm yazdırırsa kurulum tamam  
   
**İndirme başarısız olursa** (proxy/internet kısıtı): tgz dosyasını kendi  
   
 bilgisayarına indirip arayüze kopyala:  
# KENDİ bilgisayarında (VPN açıkken):  
 scp ollama-linux-amd64.tgz muhtan@172.16.6.11:~/ollama/  
   
Sonra arayüzde yukarıdaki tar -xzf adımını uygula.  
PATH satırını her seferinde yazmamak için ~/.bashrc sonuna ekleyebilirsin:  
echo 'export PATH=$HOME/ollama/bin:$PATH' >> ~/.bashrc  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSPBCj7fFRYQwYwEZiywEZJWQZeZ2ao9AAD+4lyruzq+ngAA8Nr1AMTJBeJDClAyAAAAAElFTkSuQmCC)  
**3. Modeli indir (arayüzde — çünkü node'da internet yok)**  
Modelleri scratch alanına indirelim (yazılabilir + hızlı I/O).  
   
 **Not:** scratch geçici alandır, 30 gün dokunulmazsa silinebilir.  
   
 Kalıcı tutacaksan ev dizinini kullan (kota elverirse).  
export OLLAMA_MODELS=/arf/scratch/$USER/ollama/models  
 mkdir -p "$OLLAMA_MODELS"  
 export OLLAMA_HOST=127.0.0.1:11434  
 export PATH=$HOME/ollama/bin:$PATH  
   
 # Sunucuyu arka planda başlat (indirme için; ağır iş değil, sadece download)  
 ollama serve > /tmp/ollama_pull.log 2>&1 &  
 sleep 5  
   
 # İstediğin modelleri çek (örnekler):  
 ollama pull llama3.2:3b  
 ollama pull qwen2.5:7b  
 # ollama pull llama3.1:8b  
 # ollama pull gemma2:9b  
   
 ollama list           # indirilenleri göster  
 kill %1               # arayüzdeki geçici sunucuyu durdur  
   
*OLLAMA_MODELS* * yolunu aklında tut — Adım 4'teki betikte * ***aynı yolu***  
 *  
 kullanacaksın, yoksa "model bulunamadı" hatası alırsın.*  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhZkcWinidKEAGFtgISaugy8zs1RkAAH9xr9VWHV9PAAB47XoArHwEP2w8TQAAAAAASUVORK5CYII=)  
**4. Inference'i GPU node'unda çalıştır (asıl iş)**  
**Yöntem A — Batch betiği (önerilen)**  
Arayüzde ollama_test.slurm adında dosya oluştur:  
#!/bin/bash  
 #SBATCH -p akya-cuda           # GPU kuyruğu  
 #SBATCH -A muhtan              # hesap adın (Adım 0a ile doğrula)  
 #SBATCH -J ollama-test         # iş adı  
 #SBATCH -o ollama-%j.out       # çıktı dosyası (%j = iş no)  
 #SBATCH --gres=gpu:1           # 1 GPU iste (ZORUNLU)  
 #SBATCH -N 1  
 #SBATCH -n 1  
 #SBATCH -c 10                  # akya-cuda: GPU başına 10 çekirdek  
 #SBATCH --time=01:00:00        # süre sınırı (vermezsen iş 2 dk'da kesilir)  
   
 export PATH=$HOME/ollama/bin:$PATH  
 export OLLAMA_MODELS=/arf/scratch/$USER/ollama/models   # Adım 3 ile AYNI yol  
 export OLLAMA_HOST=127.0.0.1:11434  
   
 echo "=== GPU kontrol ==="  
 nvidia-smi  
   
 echo "=== Ollama sunucusu başlatılıyor ==="  
 ollama serve > ollama_server-$SLURM_JOB_ID.log 2>&1 &  
 SERVER_PID=$!  
 sleep 10                       # sunucunun açılmasını bekle  
   
 echo "=== Modeller ==="  
 ollama list  
   
 echo "=== Inference testi ==="  
 ollama run qwen2.5:7b "Türkiye'nin başkenti neresidir? Tek cümleyle açıkla."  
   
 # Alternatif: HTTP API ile (programatik kullanım)  
 # curl -s http://127.0.0.1:11434/api/generate \  
 #   -d '{"model":"qwen2.5:7b","prompt":"Merhaba, kendini tanıt.","stream":false}'  
   
 echo "=== Sunucu kapatılıyor ==="  
 kill $SERVER_PID  
   
Gönder ve takip et:  
sbatch ollama_test.slurm     # "Submitted batch job 12345" döner  
 squeue -u $USER              # PD=bekliyor, R=çalışıyor  
 # bittiğinde çıktıyı oku:  
 cat ollama-12345.out  
   
İşi iptal etmek için: scancel 12345  
**Yöntem B — İnteraktif (canlı sohbet / deneme için)**  
GPU node'unda interaktif kabuk al:  
srun -p akya-cuda -A muhtan --gres=gpu:1 -N 1 -n 1 -c 10 \  
      --time=01:00:00 --pty bash -i  
   
Sıra gelip node'a düşünce:  
export PATH=$HOME/ollama/bin:$PATH  
 export OLLAMA_MODELS=/arf/scratch/$USER/ollama/models  
 export OLLAMA_HOST=127.0.0.1:11434  
   
 nvidia-smi  
 ollama serve > ~/ollama_int.log 2>&1 &  
 sleep 10  
 ollama run qwen2.5:7b          # interaktif sohbet; çıkmak için /bye  
   
Çıkmak için sohbet ekranında /bye, sonra node'dan çıkmak için exit.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OQQmAABRAsSdYxKa/jL0MIR7FCt5E2BJsmZmt2gMA4C+Otbqr8+sJAACvXQ85SAYUQNBTfQAAAABJRU5ErkJggg==)  
**5. GPU gerçekten kullanılıyor mu? (doğrulama)**  
İş çalışırken kontrol için:  
- Çıktıdaki nvidia-smi bölümünde GPU görünmeli.  
- Model yüklendikten sonra ollama ps GPU'da yüklü modeli gösterir.  
- nvidia-smi çıktısında ollama/runner süreci VRAM kullanıyor olmalı.  
İnteraktif oturumdaysan ikinci bir komutla:  
watch -n 2 nvidia-smi  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhwgJGkPcrHpnRgQU2QtIq6DIze3UGAMBf3Gu1VcfXEwAAXrseaJkELjbMzy0AAAAASUVORK5CYII=)  
**6. Sık karşılaşılan sorunlar**  
| | |  
|-|-|  
| **Belirti** | **Sebep / Çözüm** |   
| İş hemen reddediliyor | GPU kuyruğunda --gres=gpu:1 eksik veya yanlış -A hesap adı. |   
| model not found | Betikteki OLLAMA_MODELS yolu, Adım 3'te indirdiğin yolla aynı değil. |   
| ollama pull node'da takılıyor/hata | Beklenen: hesaplama node'larında internet yok. Modeli **arayüzde** indir. |   
| İş 2 dakikada bitiyor | #SBATCH --time=... satırını koymamışsın. |   
| GPU görünmüyor (nvidia-smi boş) | CPU kuyruğuna düşmüşsün ya da --gres yok; akya-cuda + --gres=gpu:1 kullan. |   
| address already in use (port) | OLLAMA_HOST=127.0.0.1:11435 gibi başka port ver. |   
| Model VRAM'a sığmıyor | Daha küçük model / daha düşük kuantizasyon (q4) seç ya da --gres=gpu:2 dene. |   
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OQQmAABRAsSfYxZo/khWsYQLPJrCCNxG2BFtmZquOAAD4i3Ot7mr/egIAwGvXA4qjBdKlX6OKAAAAAElFTkSuQmCC)  
**7. Hızlı kopyala-yapıştır akışı (özet)**  
# --- ARAYÜZDE (172.16.6.11), bir kez kurulum + model indirme ---  
 cd $HOME && mkdir -p ollama && cd ollama  
 curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama.tgz  
 tar -xzf ollama.tgz && rm ollama.tgz  
 export PATH=$HOME/ollama/bin:$PATH  
 export OLLAMA_MODELS=/arf/scratch/$USER/ollama/models  
 mkdir -p "$OLLAMA_MODELS"  
 ollama serve >/tmp/ol.log 2>&1 & sleep 5  
 ollama pull qwen2.5:7b  
 ollama list  
 kill %1  
   
 # --- iş gönder ---  
 sbatch ollama_test.slurm  
 squeue -u $USER  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAM0lEQVR4nO3OMQ0AIAwAwZKQ6kBqjSAOJywYYCIkd9OP36pqRMQMAAB+sfqJfLoBAMCN3NYoAzBA+QG0AAAAAElFTkSuQmCC)  
**Notlar**  
- Çalışma dosyaların ve modellerin /arf/home/$USER veya /arf/scratch/$USER  
   
 altında olmalı. scratch hızlıdır ama geçicidir (kalıcı depolama değildir).  
- Bu kılavuz açık erişimli **akya-cuda** içindir. palamut-cuda / kolyoz-cuda  
   
 erişimin varsa, ayrı arayüz **cuda-ui (172.16.6.16)** üzerinden çalışırsın ve  
   
 GPU başına 16 çekirdek istersin.  
- Takıldığın yerde TRUBA teknik destek: [grid-teknik@ulakbim.gov.tr](mailto:grid-teknik@ulakbim.gov.tr "mailto:grid-teknik@ulakbim.gov.tr")  
