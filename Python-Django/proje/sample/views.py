
from django.shortcuts import render, HttpResponse
from .forms import ImageForm
from .models import Image
from django.contrib import messages
from django.core.files.storage import FileSystemStorage

from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from keras.applications.xception import preprocess_input
import os

model= load_model('./models/thebest_model.h5')

# Create your views here.

def index(request):
    form  = ImageForm()
    img = Image.objects.all()

    return render(request,'index.html',{'img':img,'form':form})


def predictimage(request):

    import time
    t0 = time.time()
    form  = ImageForm(request.POST,request.FILES)
    
    if form.is_valid():
        file = request.FILES["photo"]
        fs = FileSystemStorage()
        if file.size > 20971520 :
            messages.error(request,"Byte boyutunu aştığınız için kaydedilmedi")
            colors1 = "red"
            return render(request,'index.html',{'form':form, 'colors1':colors1})
        
        else:
            form.save()
            filepathname = fs.save(file.name,file)
            messages.success(request,"Dosya Başarılı Bir Şekilde Yüklendi...")
            colors1 = "green"

    else:
        messages.error(request, 'Dosyayı istenilen imajda yükleyiniz.(.png,.jpeg,.jpg)')
        colors1 = "red"
        return render(request,'index.html',{'form':form, 'colors1':colors1})



    from imageio import imread
    from skimage.transform import resize
    import cv2
    from cv2 import COLOR_BGR2GRAY

    filepathname = fs.url(filepathname)
    print(filepathname)

    testimage = '.' + filepathname
    img = imread(testimage)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = resize(img, (96, 96))

    predi = model.predict(x.reshape(1, 96, 96, 1))
    t1 = time.time()
    elapsed = (t1-t0)
    tahmin_zaman= time.strftime("%H:%M:%S:{}".format(str(elapsed % 1)[15:])[:11], time.gmtime(elapsed))
    tahmin_zaman2 = tahmin_zaman.split(":")[3]
    import numpy as np
    predictedclass = str(predi.argmax())
    print(predictedclass)
 
    from decimal import Decimal

    predicted_probability = predi.max()
    predicted_probability = round(predicted_probability,2)
    predicted_probability = predicted_probability*100
    predicted_probability = "{:.0f}".format(predicted_probability)

    deneme = predi[0][0]
    deneme2= predi[0][1]
    print(deneme)
    print(deneme2)
    
    if predictedclass == "0":
        result = "Tümör Yok"
        color="green"
    elif (predictedclass == "1"):
        result = "Tümörlü" 
        color="red"  
    else:
        result = "hatalı"
        color="yellow"


######################################FEATUREMAPS#############################
    folder_name= filepathname.split("/")[2].split(".")[0]
    os.mkdir(f"media/savefigures/{folder_name}/")



    import matplotlib.pyplot as plt


    layers = [1,3,5,7]
    from keras.models import Model
    for i in layers:
        model2 = Model(inputs=model.inputs , outputs=model.layers[i].output)
        print()
        print(f"##############################################{model2.layers[i].name}#################################################\n")
        features = model2.predict(x.reshape(1, 96, 96, 1))

        fig = plt.figure(figsize=(10,7))
        for j in range(1,features.shape[3]+1):
            plt.subplot(4,8,j)
            plt.imshow(features[0,:,:,j-1] , cmap='gray')

        plt.savefig(f"media/savefigures/{folder_name}/test_{i}.png")

    
  #plt.show()

    conv_1= f"/media/savefigures/{folder_name}/test_1.png"
    conv_3= f"/media/savefigures/{folder_name}/test_3.png"
    conv_5= f"/media/savefigures/{folder_name}/test_5.png"
    conv_7= f"/media/savefigures/{folder_name}/test_7.png"
  
    print(conv_1)
    print(conv_3)
    print(conv_5)
    print(conv_7)


    tahmin = "Tahmin Sonucu : "  
    yüzde = "%"
    oran = "  oranında eşleşme"  
    part1 ="Modelin bu girdiyi tahmin etmesi "
    part2 = "milisaniye"
    part3 = " sürmüştür."
    adimlar = " Beyin tümörü tespitinde gerçekleşen adımlar :"
    cikti_1 ="1.konvolüsyon çıktısı"
    cikti_2 ="2.konvolüsyon çıktısı"
    cikti_3 ="3.konvolüsyon çıktısı"
    cikti_4 ="4.konvolüsyon çıktısı"

    context= {
        'color':color,
        'yüzde': yüzde,
        'part1' : part1 ,
        'part2' : part2 ,
        'part3' :part3,
        'oran' : oran,
        'tahmin' : tahmin,
        'filepathname': filepathname,
        'result': result,
        'predicted_probability' : predicted_probability,
        'tahmin_zaman': tahmin_zaman2,
        'conv_1':conv_1,
        'conv_3':conv_3,
        'conv_5':conv_5,
        'conv_7':conv_7,
        'adimlar': adimlar,
        'cikti_1' : cikti_1,
        'cikti_2' : cikti_2,
        'cikti_3' : cikti_3,
        'cikti_4' : cikti_4,
        'colors1' : colors1

           }


    

    return render(request,'index.html',context)



    
