### This is mockup pipeline for an image classification task that will reside on an NVIDIA Jetson device 

This pipeline proceeds by sequentially processing each JPG/PNG image in four primary steps:

1) [YOLOv5](https://github.com/ultralytics/yolov5) object detection algorithm to identify motor vehicles (i.e. a car, truck, or bus) in an image
    - Generate bounding box coordinates for all vehicles
    - Restrict to the largest vehicle (in square pixels) if multiple vehicles are detected
    - Dilate the bounding box by 5 pixels, since the bounding boxes output by YOLOv5 tend to be very tight
<br> <br />
2) Crop and resize the image array
    - The image array is cropped to the dilated bounding box coordinates
    - Resize the cropped image array to 224 x 224 pixels to adhere to the dimension of the pretrained ResNet50 model in step #3
    - This portion of the pipeline is done using a TensorFlow dataset, though the image is converted back to a numpy array for step #3
<br> <br />
3) Feed the image array to the vehicle [make-model classifier](https://github.com/kingjosephm/vehicle_detection_make_model_classifier)
    - This model was developed and trained as a TensorFlow Keras model
    - The final model weights were then converted to Onnx format
    - This model outputs a numpy array of softmax probabilities equal to the number of classes (n=574). Taking the argmax, we obtain the predicted vehicle make-model class
<br> <br />
4) Affix the bounding box and make-model label to the original image array and output as a PNG
   - Examples:

![Cadillac](<./examples/Cadillac XT6_b61697ed2a.jpg>)

![Lincoln](<./examples/Lincoln Corsair_384a67bcc4.jpg>)

![Mazda](<./examples/Mazda 626_4f203af936.jpg>)