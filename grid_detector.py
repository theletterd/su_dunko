import cv2
import numpy
from scipy import stats
import vectoriser
import sys
from collections import Counter

def extract_grid(img):

    # maybe we should scale this to a particular size.
    #img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = img.shape[:2]

    new_width = 800
    factor = float(new_width) / width
    new_height = int(height * factor)

    img = cv2.resize(img, (new_width, new_height))

    ## lets take 5 px off all sides
    #img = img[5:-5, 5:-5]
    original_img = img.copy()

    img = cv2.blur(img, (8, 8))

    # threshold image
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    #write_img(img)
    #kernel = numpy.ones((5,5))
    #img = cv2.erode(img, kernel)

    # find all the contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contoured_img = cv2.drawContours(original_img, contours, -1, (255,255,255), 5)
    #write_img(contoured_img, "hi.jpg")


    # need to find the largest, most square contour. Grrr. This is going to be hard.


    parents = hierarchy[0].transpose()[3]
    counter = Counter(parents)
    sorted_parents = [
        parent for parent, count in
        sorted(counter.items(), key=lambda i: i[1], reverse=True)
    ]

    for parent in sorted_parents:
    # find the contour that contains the most others - is probably the outside box
        container = contours[parent]

        # quad
        epsilon = 0.1 * cv2.arcLength(container, True)
        container = cv2.approxPolyDP(container, epsilon, True)
        if len(container) == 4:
            break

    # there's a better way of doing this, I'm sure.
    points = []
    for point in container:
        x, y = point[0]
        points.append([x, y])

    sum_of_values = [x + y for x, y in points]
    index_of_min = min(enumerate(sum_of_values), key=lambda i: i[1])[0]
    cycled_points = points[index_of_min:] + points[:index_of_min]

    input_points = numpy.float32(cycled_points)

    MAX_DIMENSION = 28 * 9 * 4

    output_points = numpy.float32([[0, 0], [MAX_DIMENSION, 0], [MAX_DIMENSION, MAX_DIMENSION], [0, MAX_DIMENSION]])
    try:
        transform = cv2.getPerspectiveTransform(input_points, output_points)

        # get only the grid! :O
        img = cv2.warpPerspective(original_img, transform, (MAX_DIMENSION, MAX_DIMENSION))
    except:
        pass
    return img


def get_blob_detector():
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;

    params.filterByColor = True
    params.blobColor = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 100000

    params.minRepeatability = 4
    # Filter by Circularity
    params.filterByCircularity = False

    # Filter by Convexity
    params.filterByConvexity = False
    #params.minConvexity = 0.87

    # Filter by Inertia
    #params.filterByInertia = False
    #params.minInertiaRatio = 0.01
    detector = cv2.SimpleBlobDetector_create(params)
    return detector



def process_image_for_blob_detection(img):
    original_img = img.copy()
    inverted_img = cv2.blur(img, (8, 8))
    inverted_img = cv2.adaptiveThreshold(
        inverted_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )
    inverted_img = (inverted_img * -1) + 255
    inverted_img = inverted_img.astype('uint8')



    kernel = numpy.ones((4, 4))
    inverted_img = cv2.morphologyEx(inverted_img, cv2.MORPH_OPEN, kernel)


    factor = 4
    inverted_img = cv2.resize(inverted_img, (28 * 9 * factor, 28 * 9 * factor))
    inverted_img = remove_grid(inverted_img, 28 * 4 * factor)
    inverted_img = cv2.morphologyEx(inverted_img, cv2.MORPH_OPEN, kernel)
    inverted_img = cv2.resize(inverted_img, (28 * 9, 28 * 9))
    kernel = numpy.ones((2, 2))
    inverted_img = cv2.erode(inverted_img, kernel)

    #inverted_img = cv2.blur(inverted_img, (10, 10))

    # what if we draw the grid in black on it?
    #for i in xrange(10):
    #    cv2.line(inverted_img, ((28 * i), 0), ((28 * i), (28*9) - 1), 0, 3)

    #for i in xrange(10):
    #    cv2.line(inverted_img, (0, (28 * i)), ((28*9) - 1, (28 * i)), 0, 3)

    #cv2.line(inverted_img, (0, 0), (0, (28*9) - 1), 0, 3)

    #kernel = numpy.ones((2, 2))
    #inverted_img = cv2.morphologyEx(inverted_img, cv2.MORPH_OPEN, kernel)
    #kernel = numpy.ones((2, 2))
    #inverted_img = cv2.erode(inverted_img, kernel)
    #inverted_img = cv2.erode(inverted_img, kernel)
    ##inverted_img = cv2.morphologyEx(inverted_img, cv2.MORPH_OPEN, kernel)


    # Line detection/removal.
    if False:
        edges = cv2.Canny(inverted_img, 0, 255, apertureSize=5)
        write_img(edges)

        lines = cv2.HoughLines(edges, 1, numpy.pi/180, 200)
        for line in lines:
            for rho, theta in line:
                a = numpy.cos(theta)
                b = numpy.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(inverted_img, (x1,y1), (x2,y2), 255, 4)

    return inverted_img



def get_vector_by_coords(x, y, image):
    x_start = x * 28
    y_start = y * 28
    x_end = x_start + 28
    y_end = y_start + 28
    return image[x_start:x_end, y_start:y_end]


def print_vector(vector):
    print vectoriser.print_vector(vector.astype('uint8'))


def predict_vector(x, y, img, detector, classifier):
    inverted_img = img.copy()
    vector = get_vector_by_coords(x, y, inverted_img)
    keypoints = detector.detect(vector)
    if keypoints:
        # what, I can get the minimum bounding box by using floodfill?!
        point = int(round(keypoints[0].pt[0])), int(round(keypoints[0].pt[1]))

        # we need to find the closest point which has a non-zero value.
        # stupid.
        mask = numpy.zeros((30, 30), numpy.uint8)
        old_vector = vector.copy()
        _, _, _, rect = cv2.floodFill(vector, mask,  point, 255)

        input_points = numpy.float32([
            [rect[0], rect[1]],
            [rect[0], rect[1] + rect[3]],
            [rect[0] + rect[2], rect[1] + rect[3]],
            [rect[0] + rect[2], rect[1]]
        ])
        border = 0
        output_points = numpy.float32([
            [border, border],
            [border, 28 - border],
            [28 - border, 28 - border],
            [28 - border, border]
        ])
        transform = cv2.getPerspectiveTransform(input_points, output_points)

        new_vector = cv2.warpPerspective(vector, transform, (28, 28))
        kernel = numpy.ones((2, 2))
        new_vector = cv2.erode(new_vector, kernel)
        new_vector = cv2.erode(new_vector, kernel)

        return str(classifier.predict(new_vector.reshape(1, -1))[0])
    else:
        return ' '

def predict_vector_2(x, y, img, classifier):
    vector = get_vector_by_coords(x, y, img)
    new_vector = contains_number(vector)
    if new_vector != None:
        return str(classifier.predict(new_vector.reshape(1, -1))[0])
    else:
        return ' '


def print_predicted_grid(img, classifier):
    for row in xrange(9):
        rowwww = []
        for column in xrange(9):
            #rowwww.append(predict_vector(row, column, img, detector, classifier))
            rowwww.append(predict_vector_2(row, column, img, classifier))
        print ' '.join(rowwww)


def write_img(img, filename="hi.jpg"):
    cv2.imwrite(filename, img)

#def write_vector_to_file(digit, vector):

def contains_number(img):
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    max_contour = max(contours, key=lambda x: cv2.contourArea(x))
    if cv2.contourArea(max_contour) < 10:
        return None

    # ok, so now we get the rectangle... and scale it.
    x, y, w, h = cv2.boundingRect(max_contour)
    input_points = numpy.float32([
        [x, y],
        [x, y + h],
        [x + w, y + h],
        [x + w, y]
    ])

    output_points = numpy.float32([
        [0,0],
        [0, 28],
        [28, 28],
        [28, 0],
    ])
    transform = cv2.getPerspectiveTransform(input_points, output_points)
    new_vector = cv2.warpPerspective(img, transform, (28, 28))

    return new_vector * 255


def write_vectors(image):
    for i in xrange(9):
        for j in xrange(9):
            vector = get_vector_by_coords(i, j, image)
            new_vector = contains_number(vector)
            if new_vector != None:
                print_vector(new_vector)
                raw_vector = new_vector.reshape(1, -1)[0]
                try:
                    character = input()
                    if character in {1, 2, 3, 4, 5, 6, 7, 8, 9}:
                        with open('./data/mydata/data' + str(character), 'a') as f:
                            f.write(bytearray(raw_vector))
                    sys.exit()
                except:
                    pass



def remove_grid(img, line_length):
    line_width = 1

    h_lines = img.copy()
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, line_width));
    h_lines = cv2.erode(h_lines, horizontalStructure)
    h_lines = cv2.dilate(h_lines, horizontalStructure)

    v_lines = img.copy()
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (line_width, line_length));
    v_lines = cv2.erode(v_lines, verticalStructure)
    v_lines = cv2.dilate(v_lines, verticalStructure)
    #write_img(img, "hi.jpg")

    gridless_img = img - (v_lines + h_lines)
    return gridless_img


#detector = get_blob_detector()
#img = extract_grid('./puzzles/testing/sudoku{}.jpg'.format(sys.argv[1]))
#img = process_image_for_blob_detection(img)


cam = cv2.VideoCapture(0)

while True:
    val, cam_img = cam.read()
    img = extract_grid(cam_img)

    cv2.imshow('webcam', cam_img)
    cv2.imshow('grid', img)
    if cv2.waitKey(1) == 27: #esc
        break

cv2.destroyAllWindows()

img_copy = img.copy()
#keypoints = detector.detect(img)
classifier = vectoriser.get_trained_classifier("./data/mydata/")
print_predicted_grid(img, classifier)

#import ipdb; ipdb.set_trace()
#img = cv2.drawContours(original_img, [container], 0, (255,255,255), 10)
#import ipdb; ipdb.set_trace()

# img = cv2.drawKeypoints(
#     img,
#     keypoints,
#     numpy.array([]),
#     (0,0,255),
#     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )
write_img(img_copy, 'hi.jpg')
#write_vectors(img)
