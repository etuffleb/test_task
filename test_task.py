import numpy as np
import cv2


# поиск области номера
# возвращает найденные вырезанные изображения 
def find_area(image):
    ims_result = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY )[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        approx = cv2.approxPolyDP(c, 0.015 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            #cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0))
            ims_result.append(image[y:y+h, x:x+w])
        else:
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            #cv2.drawContours(image, [box], 0, (0,0,255))
    return ims_result


def processing(image):
    pass

def get_numbers(image):
    pass

def main(args=None):
    
    image = cv2.imread('1.png')
    width = int(image.shape[1] * 0.5)
    height = int(image.shape[0] * 0.5)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    # первый этап
    ims_number = find_area(resized)

    for im in ims_number:
        clear_im = processing(im) # второй этап
        result = get_numbers(clear_im) # третий этап



if __name__ == '__main__':
    main()

