# Brain-Tumor-Detection-System
Görüntü işleme teknolojisini kullanarak web sayfasına girilen beyin MR görüntüsünün tümörlü olup olmadığını tahmin eden bir web asistanıdır.
![image](https://github.com/betul-000/Brain-Tumor-Detection-System/assets/75879475/f6b40d38-b7a6-4b6a-919d-de5f3b8c4213)

   Beyin tümörü, her yaştan insanı etkileyen ve ölümlere sebep olan en yaygın rahatsızlıklardan birisidir. Beyin tümörünün erken ve doğru teşhisi insan hayatı için oldukça önemlidir. Teknolojinin gelişmesiyle birlikte Yapay Zeka teknolojisi sağlık alanında aktif kullanılarak hastalıkların hızlı ve doğru şekilde tespit edilmesini sağlamaktadır. Hastalığın görüntülenmesinde kullanılan MR (Manyetik Rezonans) görüntüleri beyin tümörü rahatsızlığı için en doğru ve etkili görüntüleri sağlayan araçtır. Son yıllarda yaygınlaşan derin öğrenme teknikleri görüntülerin işlenmesinde sıklıkla kullanılmaktadır. Derin öğrenme tekniklerini kullanarak MR görüntülerinden beyin tümörü tespiti uygulamasını gerçekleştirmek sağlık çalışanlarına zaman kazandırırken hastaların erken teşhis sayesinde hastalıktan kurtulma oranlarını artırmaktadır. Bu çalışmada beyin tümörü tespiti için 𝑀𝑜𝑏𝑖𝑙𝑒𝑁𝑒𝑡𝑉2, 𝑋𝑐𝑒𝑝𝑡𝑖𝑜𝑛, 𝐼𝑛𝑐𝑒𝑝𝑡𝑖𝑜𝑛𝑉3, 𝑅𝑒𝑠𝑁𝑒𝑡152𝑉2, 𝐼𝑛𝑐𝑒𝑝𝑡𝑖𝑜𝑛𝑅𝑒𝑠𝑁𝑒𝑡𝑉2, 𝐷𝑒𝑛𝑠𝑒𝑁𝑒𝑡121 ve özgün CNN modeli önerilmiştir. Bu modeller ile tümör tespiti gerçekleştirilmiş ve modeller bir altın standart veriseti üzerinde değerlendirilmiştir. Sonuçların başarı metrik değerleri elde edilmiş ve başarı değerleri karşılaştırılmıştır. Deneysel sonuçlara göre öne sürülen özgün model %98.7’lik doğruluk elde etmiştir. 
  
 Özgün model web sayfasına entegre edilmiştir. Tahmin edilmesi istenen MR görüntülerin yüklenmesinden sonra beyinin tümörlü veya tümörsüz olduğunu ve hangi olasılıkla bu sonuca ulaştığı kullanıcı arayüzünde belirtilmiştir. Bunun dışında yüklenilen resmin model tarafından tahmin yaptırılırken hangi özelliklere göre tahmin yaptığını gösteren özellik haritaların (Feature Maps) çıktısını da kullanıcı arayüzünde verilmiştir.
    

  ![image](https://github.com/betul-000/Brain-Tumor-Detection-System/assets/75879475/2be3ab50-2247-4710-be2d-f52cfeadc87d)

