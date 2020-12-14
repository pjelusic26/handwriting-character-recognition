
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter

import pytesseract
from pytesseract import Output

import tensorflow as tf

import operator

class Photomath:

    @staticmethod
    def imgPreprocess(filename):
        """Method for importing and processing images

        Args:
            filename (string): Path to file

        Returns:
            ndarray: Processed image
        """

        cv2_img = np.array(Image.open(filename))
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        return thresh

    @staticmethod
    def recognizeCharacters(image):
        """Method for recognizing characters from image

        Args:
            image (ndarray): Input image

        Returns:
            (string): Recognized text and data (coordinates)
        """

        cv2_text = pytesseract.image_to_string(image, lang = 'eng', config = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789+-')
        cv2_data = pytesseract.image_to_boxes(image)

        return cv2_text, cv2_data

    @staticmethod
    def saveCharacters(image, image_orig):
        """Method for writing recognized characters as individual images

        Args:
            image (ndarray): processed image
            image_orig (ndarray): original image
        """

        image_orig = np.array(Image.open(image_orig))
        image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)

        # Draw bounding boxes
        cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        counter = 0

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

            img_crop = image_orig[y: y + h, x: x + w]

            cv2.imwrite(f"testset/recognized_character_{counter}.jpg", img_crop)

            counter += 1

    @staticmethod
    def squareImage(image, min_size = 28, fill_color = (0)):
        """Method for making an image square

        Args:
            image (ndarray): original character image
            min_size (int, optional): Minimal axis dimension of image. Defaults to 28.
            fill_color (tuple, optional): Colour to fill around original character image. Defaults to (0).

        Returns:
            (ndarray): Square image
        """

        im = Image.open(image)
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        new_image = new_im.resize((min_size, min_size))
        return new_image

    @staticmethod
    def showImage(image, figsize, title):
        """Method for inline displaying of image

        Args:
            image (ndarray): Input image
            figsize (int): Dimension of the figure
            title (string): Plot title
        """

        fig = plt.figure(figsize = (figsize, figsize))
        fig.suptitle(title, fontsize = 16)
        plt.imshow(image, cmap = "gray")

    def writeImage(filename, image):
        """Method for image writing

        Args:
            filename (string): Image path and name
            image (ndarray): input image
        """

        cv2.imwrite(filename, image)

    @staticmethod
    def fitMnist(trainSize):
        """Method for training neural net

        Args:
            trainSize (int): Decides how big the training size is

        Returns:
            (keras model): Trained model
        """

        # Load mnist
        mnist = tf.keras.datasets.mnist
        (x_train, y_train),(x_test, y_test) = mnist.load_data()

        # Slicing train dataset
        x_train = x_train[0: trainSize, :, :]
        y_train = y_train[0: trainSize]

        # Normalize the train dataset
        x_train = tf.keras.utils.normalize(x_train, axis = 1)

        # Normalize the test dataset
        x_test = tf.keras.utils.normalize(x_test, axis = 1)

        #Build the model object
        model = tf.keras.models.Sequential()

        # Add the Flatten Layer
        model.add(tf.keras.layers.Flatten())

        # Build the input and the hidden layers
        model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))

        # Build the output layer
        model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

        # Compile the model
        model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

        # Fit model
        model.fit(x = x_train, y = y_train, epochs = 5, verbose = 0)

        # Evaluate the model performance
        test_loss, test_acc = model.evaluate(x = x_test, y = y_test, verbose = 0)

        # Print out the model accuracy
        print('\nTest accuracy:', test_acc)

        return model

    @staticmethod
    def importImages(path_to_file_1, path_to_file_2, path_to_file_3):
        """Method for importing three images

        Args:
            path_to_file_1 (string): Path to image
            path_to_file_2 (string): Path to image
            path_to_file_3 (string): Path to image

        Returns:
            (ndarray): Stacked array of images
        """

        image_1 = np.asarray(Image.open(path_to_file_1))
        image_2 = np.asarray(Image.open(path_to_file_2))
        image_3 = np.asarray(Image.open(path_to_file_3))

        stacked = np.stack((image_1, image_2, image_3))

        return stacked

    @staticmethod
    def getPredictions(model, images):
        """Method for getting neural net predictions

        Args:
            model (keras model): Trained model
            images (ndarray): Images to predict

        Returns:
            (int): Prediction
        """

        predictions = model.predict([images])

        return np.argmax(predictions[0]), np.argmax(predictions[1]), np.argmax(predictions[2])

    @staticmethod
    def divideString(string):
        """Method for dividing string to digits and operators

        Args:
            (string): Input string

        Returns:
            (list): Lists of digits and operators
        """
    
        ops = {
            '+' : operator.add,
            '-' : operator.sub,
            '*' : operator.mul,
            '/' : operator.truediv,
            '%' : operator.mod,
            '^' : operator.xor,
            '=' : operator.eq,
        }

        digits = []
        operators = []
        for i in string:
            if i.isdigit() == True:
                digits.append(int(i))
            elif i.isdigit() == False:
                if i in ops.keys():
                    operators.append(i)
                else:
                    pass

        return digits, operators

    @staticmethod
    def result(digits, operators):
        """Method for calculating mathematical expression

        Args:
            digits (list): List of digits
            operators (list): List of operators

        Returns:
            (int): Numeric result
        """
        
        ops = {
            '+' : operator.add,
            '-' : operator.sub,
            '*' : operator.mul,
            '/' : operator.truediv,
            '%' : operator.mod,
            '^' : operator.xor,
            '=' : operator.eq,
        }
    
        result_temp = 0

        while len(digits) >= 2:

            result_temp = ops[operators.pop(0)](digits.pop(0), digits.pop(0))
            digits.insert(0, result_temp)

        return int(digits[0])