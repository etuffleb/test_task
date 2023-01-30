from matplotlib import pyplot as plt
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY )[1]
    edges = cv2.Canny(thresh, 150, 150)

    lines_list =[]
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=5, maxLineGap=5 )
    if lines is None:
        return thresh
  
    for points in lines:
        x1,y1,x2,y2=points[0]
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),1)
        lines_list.append([(x1,y1),(x2,y2)])

    return thresh


def get_numbers(image):
    hist = np.sum(image, axis = 0) / image.shape[0]    
    plt.plot(hist)
    plt.xlim([0, image.shape[1]])
    plt.show()

    hist_list = hist.tolist()
    max_i = max(hist_list)
    
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)    
    i = 0
    for ind in hist_list:
        if ind == max_i:
            cv2.line(image, (i,0), (i,image.shape[0]), (0,255,0))
        i += 1
    # cv2.imshow('image2', image)
    # cv2.waitKey()
    

def main(args=None):    
    image = cv2.imread('1.png')
    # первый этап
    ims_number = find_area(image)
    for im in ims_number:
        clear_im = processing(im) # второй этап
        result = get_numbers(clear_im) # третий этап


if __name__ == '__main__':
    main()

