import cv2
import numpy as np
# from skimage import iradon
from PIL import Image
from scipy import signal

lines = []
sinogram = []


def radon_transform(image_name, number_of_emitters=180, angular_step=1, angular_range=180):
    image = cv2.imread(image_name, 0)

    image = pad_image(image)
    print(image.shape)

    ANGLES = np.arange(0, angular_range, angular_step)
    OFFSETS = np.linspace(-image.shape[0] / 2, image.shape[0] / 2, number_of_emitters)
    radius = image.shape[0] / 2

    for angle in ANGLES:
        angle = np.deg2rad(angle)
        sinogram_row = []
        lines_row = []
        for offset in OFFSETS:
            positions = get_positions(radius, angle, offset)
            single_line = create_line(radius, positions)
            color = calculate_single_line(image, single_line)
            sinogram_row.append(color)
            lines_row.append(single_line)
        sinogram.append(sinogram_row)
        lines.append(lines_row)

    # print(lines)
    # print(sinogram)

    image = Image.fromarray(np.array(sinogram))
    image.show()
    # cv2.imwrite("RadonSheppchuj.jpg", np.array(sinogram))
    return np.array(sinogram)

    # cv2.imshow('tomograf', image)

    # cv2.waitKey(0)


# funkcja wyrownujaca prostokatne zdjecie do kwadratowego
def pad_image(image):
    width, height = np.shape(image)

    if width > height:
        difference = (width - height) // 2
        image = cv2.copyMakeBorder(image, 0, 0, difference, difference, cv2.BORDER_CONSTANT, 0)

    elif height > width:
        difference = (height - width) // 2
        image = cv2.copyMakeBorder(image, difference, difference, 0, 0, cv2.BORDER_CONSTANT, 0)

    if image.shape[0] > image.shape[1]:
        image = cv2.copyMakeBorder(image, 0, 0, 1, 0, cv2.BORDER_CONSTANT, 0)

    elif image.shape[1] > image.shape[0]:
        image = cv2.copyMakeBorder(image, 1, 0, 0, 0, cv2.BORDER_CONSTANT, 0)

    return image


# pobieranie pozycji
def get_positions(radius, angle, offset):
    x1, y1 = radius * np.cos(angle), radius * np.sin(angle)
    x2, y2 = -x1, -y1
    xv, yv = offset * np.cos(angle + np.pi / 2), offset * np.sin(angle + np.pi / 2)
    x1 += xv + radius
    y1 += yv + radius
    x2 += xv + radius
    y2 += yv + radius
    return [x1, y1, x2, y2]


def calculate_single_line(image, single_line):
    result = []
    for x, y in single_line:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            result.append(image[x, y])
    return np.array(result).mean() if len(result) > 0 else 0


# Algorytm bresenhama do tworzenia linii
def create_line(radius, points):
    x1, y1, x2, y2 = int(points[0]), int(points[1]), int(points[2]), int(points[3])
    result = []

    kx = 1 if x1 < x2 else -1
    ky = 1 if y1 < y2 else -1

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    result.append((x1, y1))
    if dx > dy:
        e = dx / 2
        for i in range(int(dx)):
            x1 += kx
            e -= dy
            if e < 0:
                y1 += ky
                e += dx
            result.append((x1, y1))
    else:
        e = dy / 2
        for i in range(int(dy)):
            y1 += ky
            e -= dx
            if e < 0:
                x1 += kx
                e += dy
            result.append((x1, y1))

    radius_pow_2 = radius ** 2
    return [(x, y) for x, y in result if ((x - radius) ** 2 + (y - radius) ** 2) <= radius_pow_2]


# odtworzenie linii
def reverse(input_image, output_image_name="output_image.png"):
    input_image = cv2.imread(input_image, 0)

    sinogram_reverse = np.array(sinogram)

    output_image = np.zeros((input_image.shape[1], input_image.shape[0]))

    sinogram_reverse = filter_projection(sinogram_reverse)
    for i in range(sinogram_reverse.shape[0]):
        for j in range(sinogram_reverse.shape[1]):
            for x, y in lines[i][j]:
                if 0 <= x < input_image.shape[1] and 0 <= y < input_image.shape[0]:
                    output_image[x, y] += sinogram_reverse[i][j]

    image = Image.fromarray(np.array(output_image))
    image.show()
    cv2.imwrite(output_image_name, output_image)


def low_pass_filter(number_freq, value):
    filtering_array = np.full((1, number_freq * 2), value)
    return filtering_array


def hann_filter(number_freq):
    filtering_array = np.hanning(number_freq * 2)
    return filtering_array


def ramp_filter(number_freq):
    filtering_array = 2 * np.arange(number_freq + 1) / np.float32(2 * number_freq)
    filtering_array = np.concatenate((filtering_array, filtering_array[number_freq - 1:0:-1]), axis=0)
    return filtering_array


def filter_projection(sinogram):
    number_angles, number_offsets = sinogram.shape
    number_freq = 2 * int(2 ** (int(np.ceil(np.log2(number_offsets)))))

    filter_array = low_pass_filter(number_freq, 0.02)

    padded_sinogram = np.concatenate((sinogram, np.zeros((number_angles, 2 * number_freq - number_offsets))), axis=1)

    for i in range(number_angles):
        padded_sinogram[i, :] = np.real(np.fft.ifft(np.fft.fft(padded_sinogram[i, :]) * filter_array))

    return padded_sinogram[:, :number_offsets]


if __name__ == '__main__':
    radon_transform("Shepp_logan.png", 180, 1, 180)
    reverse("Shepp_logan.png")
