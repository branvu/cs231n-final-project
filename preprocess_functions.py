import cv2

FINAL_IMAGE_SIZE = 28  # want 28 x 28 image
TOTAL_WANTED = 8055  # number of images per language


def crop_img(img, margin=0, show=False, show_bounding=False):
    # create binary image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[-1] == 3 else img

    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # get contours (bounding boxes)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # should only only be one box
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if len(cnts) != 1:
        cnts = cnts[-1:]

        # display bounding boxes
        if show_bounding:
            for i, c in enumerate(cnts):
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.imshow(f"image {i}", img)
                cv2.waitKey(0)

    assert len(cnts) == 1

    # extract coords from bounding box
    x, y, w, h = cv2.boundingRect(cnts[0])

    # get square image dimensions
    edge_length = max(w, h)

    # make sure starting x and y are in bounds
    x = max(x + w//2 - edge_length//2 - margin, 0)
    y = max(y + h//2 - edge_length//2 - margin, 0)

    if show:
        cv2.imshow("Original image", img)
        cv2.waitKey(0)

    # crop image
    end_x = min(x + edge_length + margin, img.shape[1])
    end_y = min(y + edge_length + margin, img.shape[0])

    # ensure square
    if end_x - x != end_y - y:
        edge_length = min(end_x - x, end_y - y)
    else:
        edge_length = end_x - x

    img = img[y:y+edge_length, x:x+edge_length]

    if show:
        cv2.imshow("Resized image", img)
        cv2.waitKey(0)

    return img


def resize(w, h, img):
    dim = (w, h)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def grayscale(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def show_image_progress(path):
    # display and save image showing preprocessing steps
    img = cv2.imread(path)
    cv2.imwrite("unedited.jpg", img)
    cv2.imshow("unedited", img)
    cv2.waitKey(0)

    img = crop_img(img)
    cv2.imwrite("cropped.jpg", img)
    cv2.imshow("cropped", img)
    cv2.waitKey(0)

    img = resize(FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE, img)
    cv2.imwrite("resized.jpg", img)
    cv2.imshow("resized", img)
    cv2.waitKey(0)

    img = grayscale(img)
    cv2.imwrite("gray.jpg", img)
    cv2.imshow("gray", img)
    cv2.waitKey(0)


def main():
    path = "data\isl_original_data\Person1-K-1-13.jpg"  # milestone path
    # show_image_progress(path)

    img = cv2.imread("data/arsl_original_data/ain/AIN (1).JPG")
    crop_img(img, 0, False, True)


if __name__ == "__main__":
    main()
