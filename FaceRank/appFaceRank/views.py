from django.shortcuts import render
from django.http import HttpResponse
import simplejson as json
from appFaceRank import views
from django.shortcuts import render
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2
import base64, io
from datauri import DataURI


def index(request):

    return render(request, 'appFaceRank/index.html')

    # return HttpResponse("Hello, world. You're at the polls index.")


def getPicture(request):
    
    search_key = request.GET['search_key']
    imageUrl = {'search_key': search_key}
    im = Image.open(io.BytesIO(base64.b64decode(str(imageUrl).split(',')[1])))
    im.save("ttttttsssss.jpg")
    context = {}
    #print(type(imgdata))
    #image = Image.open(context)
    #print(context)
    #sample_image = cv2.imread(context)
    
    #getImage(context)
    return HttpResponse(json.dumps(context), content_type="application/json")


def getImage(imgName):
    img = []

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    sample_image = cv2.imread('btsjpg.jpg')
    gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    bigValue = -9999999
    bigidx = 0

    idx = 0
    for (x, y, w, h) in faces:

        cv2.rectangle(sample_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        sub_face = sample_image[y:y+h, x:x+w]
        PIL_image = Image.fromarray(np.uint8(sub_face)).convert('RGB')
        #test = Image.open(PIL_image)
        # test.show()
        result = exeTeachableMachine(PIL_image)

        if bigValue < result[0][1]:
            bigValue = result[0][1]
        idx = idx+1
        #FaceFileName = "unknowfaces/face_" + str(y) + ".jpg"
        #cv2.imwrite(FaceFileName, sub_face)
    rank = idx + 1
    print('1등은 몇번째 사진인가효???'+rank)
    #cv2.imshow('sample', sample_image)
    # cv2.waitKey()
    # print(img)
    # return img


def exeTeachableMachine(image):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = tensorflow.keras.models.load_model('keras_model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # getImage()
    # Replace this with the path to your image
    #image = Image.open('sample.jpg')

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # display the resized image
    image.show()

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return prediction


if __name__ == '__main__':
    getImage('btsjpg.jpg')
    exeTeachableMachine('sample.jpg')
    # exeTeachableMachine(getImage('btsjpg.jpg'))
