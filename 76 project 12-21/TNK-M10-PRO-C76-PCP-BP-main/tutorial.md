Predict the Pneumothorax
=========================

In this activity, you will learn to detect if patient has pneumothorax using the lungs image.


<img src= "https://s3.amazonaws.com/media-p.slid.es/uploads/1525749/images/10580203/image__43_.png" width = "480" height = "220">


Follow the given steps to complete this activity:

1. Predict the pneumothorax

* Open the main.py file.

* Load the age detection model.

    `ageDetectionModel = load_model('age_model_50epochs.h5', compile=False)`

* Predict the age using age detection model by passing the `resizedImg` to it and save it in `prediction` variable.

    `prediction = ageDetectionModel.predict(resizedImg)`

* Add `prediction[0][0]` that is age of the detected face on the screen.

    `img = cv2.putText(img, str(int(prediction[0][0])), (X, Y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)`
               
* Save and run the code to check the output.

