import cv2   #image processing
import pytesseract # for OCR
import os    # for os system calls
from os import listdir  #list files on dir
import re #regex

folder_dir = "./img/"  #image directory (code_path + /img/)

for img in os.listdir(folder_dir):
    
    # find all files ending with ".png", ".jpg" or ".jpeg"
    if (img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")):
        
        # load the image and convert it to grayscale
        image = cv2.imread(folder_dir + img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # load the number plate detector
        n_plate_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
        
        # detect the number plates in the grayscale image
        detections = n_plate_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)

        # loop over the number plate bounding boxes
        for (x, y, w, h) in detections:

            # extract the number plate from the grayscale image
            number_plate = gray[y:y + h, x:x + w]
            
            cv2.imshow(img, number_plate)  #show recognized plate

            cv2.waitKey(3000)   #show image during 3 seconds
            
            cv2.destroyWindow(img) #close image frame
            
        #predict plate number from gray image using Tesseract    
        predicted_result = pytesseract.image_to_string(number_plate, lang ='eng',
        config ='--oem 3 --psm 9 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        # if predicted number is valid should contain 7 or more characters (0000XYZ, e.g.)
        if len(predicted_result) >= 7 and (predicted_result is not None):
        
            aux = re.findall("[0-9]{4}[A-Z]{3}", predicted_result) #substring containing 4 nums + 3 letters    
            
            plate_num = ''.join((aux[0][:4],' ',aux[0][4:])) #separate numbers and letters using a blank space
            
        else:
            
            plate_num = "-"  #if predicted number is not valid, print "-"

        #print image name, plate number found and valid plate number
        print("Imagen: ",img,"  Texto encontrado: ",predicted_result,"  Matrícula: ",plate_num)
        
# destroy all created windows
cv2.destroyAllWindows()

