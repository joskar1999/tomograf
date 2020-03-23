import cv2
import numpy as np
from PIL import Image as IMG
from PIL import ImageTk as IMGtk
import pydicom
import _tkinter
import tkinter
from tkinter import ttk
from tkinter import *
from functools import partial

# from pydicom.pixel_data_handlers import gdcm_handler, pillow_handler

lines = []
sinogram = []
ANGLES = []
sinogram_reverse = []
output_image = []
iteration_var = 0


def radon_transform(image_name, number_of_emitters=180, angular_step=1, angular_range=180):
    # image = cv2.imread(image_name, 0)
    image = create_pixel_array_from_dicom(image_name)
    image = pad_image(image)
    ANGLES = np.arange(0, angular_range, angular_step)
    OFFSETS = np.linspace(-image.shape[0] / 2, image.shape[0] / 2, number_of_emitters)
    radius = image.shape[0] / 2

    for angle in ANGLES:
        angle = np.deg2rad(angle)
        sinogram_row = []
        lines_row = []
        for offset in OFFSETS:
            positions = get_positions(radius, angle, offset)
            single_line = create_line(positions)
            color = calculate_single_line(image, single_line)
            sinogram_row.append(color)
            lines_row.append(single_line)
        sinogram.append(sinogram_row)
        lines.append(lines_row)

    image = IMG.fromarray(np.array(sinogram))
    image.show()

    return np.array(sinogram)


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


def create_line(points):
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

    return [(x, y) for x, y in result]


def reverse(input_image, select_filter, output_image_name="output_image.png"):
    global sinogram_reverse
    global output_image

    input_image = create_pixel_array_from_dicom(input_image)
    sinogram_reverse = np.array(sinogram)
    output_image = np.zeros((input_image.shape[1], input_image.shape[0]))
    sinogram_reverse = filter_sinogram(sinogram_reverse, select_filter)

    for i in range(sinogram_reverse.shape[0]):
        for j in range(sinogram_reverse.shape[1]):
            for x, y in lines[i][j]:
                if 0 <= x < input_image.shape[1] and 0 <= y < input_image.shape[0]:
                    output_image[x, y] += sinogram_reverse[i][j]

    image = IMG.fromarray(np.array(output_image))
    image.show()
    cv2.imwrite(output_image_name, output_image)


def initialize_before_animation(input_image, select_filter):
    global sinogram_reverse
    global output_image
    output_image = np.zeros((input_image.shape[1], input_image.shape[0]))
    sinogram_reverse = filter_sinogram(np.array(sinogram), select_filter=select_filter)


def create_photo_from_sinogram_animation(input_image):
    global iteration_var
    global sinogram_reverse
    global output_image

    if iteration_var < len(ANGLES):
        for i in range(sinogram_reverse.shape[1]):
            for x, y in lines[iteration_var][i]:
                if 0 <= x < input_image.shape[1] and 0 <= y <= input_image.shape[0]:
                    output_image[x, y] += sinogram_reverse[iteration_var][i]
        iteration_var += 1
        return True
    else:
        return False


def create_animation(input_image):
    if create_photo_from_sinogram_animation(input_image):
        image = IMG.fromarray(output_image)
        photo = IMGtk.PhotoImage(image)
        canvas_reversed_sinogram.create_image(0, 0, photo, anchor=NW)
        root.update()
        root.after(50, create_animation(input_image))


def hyperbolic_filter(number_freq):
    filtering_array = []
    for i in range(1, number_freq + 2):
        filtering_array.append(1 / i + 0.01)
    filtering_array = np.concatenate((filtering_array[number_freq - 1:0:-1], filtering_array), axis=0)
    return filtering_array / max(filtering_array)


def low_pass_filter(number_freq, value):
    filtering_array = np.full((1, number_freq * 2), value)
    return filtering_array


def hann_filter(number_freq):
    filtering_array = np.hanning(number_freq * 2)
    return filtering_array


def triangular_filter(number_freq):
    filtering_array = 2 * np.arange(number_freq + 1) / np.float32(2 * number_freq)
    filtering_array = np.concatenate((filtering_array, filtering_array[number_freq - 1:0:-1]), axis=0)
    return filtering_array


def cropped_triangular_filter(number_freq):
    filtering_array = 2 * np.arange(number_freq + 1) / np.float32(2 * number_freq)
    filtering_array[0:int(number_freq / 40)] = 0
    filtering_array = np.concatenate((filtering_array, filtering_array[number_freq - 1:0:-1]), axis=0)
    return filtering_array


def filter_sinogram(sinogram, select_filter):
    number_angles, number_offsets = sinogram.shape
    number_freq = 2 * int(2 ** (int(np.ceil(np.log2(number_offsets)))))

    if select_filter == 1:
        filter_array = hann_filter(number_freq)
    elif select_filter == 2:
        filter_array = low_pass_filter(number_freq, 0.02)
    elif select_filter == 3:
        filter_array = triangular_filter(number_freq)
    else:
        filter_array = hyperbolic_filter(number_freq)

    padded_sinogram = np.concatenate((sinogram, np.zeros((number_angles, 2 * number_freq - number_offsets))), axis=1)

    for i in range(number_angles):
        padded_sinogram[i, :] = np.real(np.fft.ifft(np.fft.fft(padded_sinogram[i, :]) * filter_array))

    return padded_sinogram[:, :number_offsets]


def create_pixel_array_from_dicom(image_name):
    if image_name.endswith('.dcm'):
        image_dcm = pydicom.dcmread(image_name)
        img = IMG.fromarray(np.uint8(image_dcm.pixel_array[0] * 255))
        # img.show()
        return np.array(img)
    return cv2.imread(image_name, 0)


def read_dicom(image_name):
    if image_name.endswith('.dcm'):
        return pydicom.dcmread(image_name)
    return None


def startSimulation():
    global sinogram
    global lines
    global sinogram_reverse

    starting_photo = []

    loaded_image = None
    image_name = image_name_placeholder.get()

    if image_name.endswith(".png"):
        starting_photo = PhotoImage(file=image_name_placeholder.get())
        loaded_image = cv2.imread(image_name, 0)
    elif image_name.endswith(".jpg"):
        starting_photo = IMG.open(image_name_placeholder.get())
        starting_photo = IMGtk.PhotoImage(starting_photo)
        loaded_image = cv2.imread(image_name, 0)
    elif image_name.endswith(".dcm"):
        loaded_image = create_pixel_array_from_dicom(image_name_placeholder.get())
        dcm_image = IMG.fromarray(loaded_image)
        starting_photo = IMGtk.PhotoImage(dcm_image)

    canvas_starting_photo.create_image(0, 0, image=starting_photo, anchor=NW)

    root.update()

    sinogram = []
    lines = []

    radon_transform(image_name_placeholder.get(), 180, 1, 180)

    sinogram_to_array = IMG.fromarray(np.array(sinogram))
    sinogram_photo = IMGtk.PhotoImage(sinogram_to_array)

    canvas_sinogram.create_image(0, 0, image=sinogram_photo, anchor=NW)
    root.update()

    # initialize_before_animation(loaded_image,select_filter=filter_mode.get())
    # create_animation(loaded_image)

    reverse(image_name_placeholder.get(), select_filter=filter_mode.get())

    reversed_sinogram_photo = IMGtk.PhotoImage(IMG.fromarray(output_image))
    canvas_reversed_sinogram.create_image(0, 0, image=reversed_sinogram_photo, anchor=NW)

    root.wait_window()


global filter_mode
global patient_name_input
global date_of_examination_input
global comment_input
global image_name_placeholder

if __name__ == '__main__':
    # Window init
    root = tkinter.Tk()
    root.resizable(0, 0)
    root.title("Tomography simulator")

    # image_name = "0002.dcm"
    loaded_image = None

    # if image_name.endswith(".png"):
    #     starting_photo = PhotoImage(file=image_name)
    #     loaded_image = cv2.imread(image_name, 0)
    # elif image_name.endswith(".jpg"):
    #     starting_photo = IMG.open(image_name)
    #     starting_photo = IMGtk.PhotoImage(starting_photo)
    #     loaded_image = cv2.imread(image_name, 0)
    # elif image_name.endswith(".dcm"):
    #     loaded_image = create_pixel_array_from_dicom(image_name)
    #     dcm_image = IMG.fromarray(loaded_image)
    #     starting_photo = IMGtk.PhotoImage(dcm_image)

    canvas_starting_photo = Canvas(root, width=300, height=300)
    canvas_starting_photo.grid(row=0, column=0)

    canvas_sinogram = Canvas(root, width=300, height=300)
    canvas_sinogram.grid(row=0, column=1)

    canvas_reversed_sinogram = Canvas(root, width=300, height=300)
    canvas_reversed_sinogram.grid(row=0, column=2)

    frame_for_inputs = Frame(root, width=250, height=300)
    frame_for_inputs.grid(row=0, column=3)
    # nie zmniejszaj okna po umieszczeniu elementu
    frame_for_inputs.grid_propagate(0)

    patient_name = StringVar()
    date_of_examination = StringVar()
    comment = StringVar()

    patient_name.set("Imię pacjenta")
    date_of_examination.set("Data badania")
    comment.set("Komentarz")

    # labelki dla inputów
    patient_name_label = Label(frame_for_inputs, textvariable=patient_name, fg="black")
    date_of_examination_label = Label(frame_for_inputs, textvariable=date_of_examination, fg="black")
    comment_label = Label(frame_for_inputs, textvariable=comment, fg="black")

    patient_name_label.grid(row=0, column=0)
    date_of_examination_label.grid(row=1, column=0)
    comment_label.grid(row=2, column=0)

    patient_name_input = Entry(frame_for_inputs)
    date_of_examination_input = Entry(frame_for_inputs)
    comment_input = Entry(frame_for_inputs)

    patient_name_input.grid(row=0, column=1)
    date_of_examination_input.grid(row=1, column=1)
    comment_input.grid(row=2, column=1)

    MODES = [
        ("Hann", 1),
        ("Low-pass", 2),
        ("Triangular", 3),
        ("Hyperbolic", 4),
    ]

    filter_mode = IntVar(frame_for_inputs)
    # filter_mode.set(3)  # initialize
    row = 3
    for text, mode in MODES:
        b = Radiobutton(frame_for_inputs, text=text,
                        variable=filter_mode, value=mode)
        b.grid(row=row, column=0)
        row = row + 1

    # canvas_starting_photo.create_image(0, 0, image=starting_photo, anchor=NW)
    # canvas_reversed_sinogram.create_image(0, 0, image=starting_photo, anchor=NW)
    # canvas_sinogram.create_image(0, 0, image=starting_photo, anchor=NW)

    # Entry dla nazwy zdjęcia
    image_name_placeholder = StringVar()
    image_name_placeholder.set(value="Nazwa zdjęcia")
    image_name_input = Entry(frame_for_inputs, textvariable=image_name_placeholder)
    image_name_input.grid(row=7, column=0)

    # przycisk startu -> dodać komand który wywoła funkcję
    startingButton = Button(frame_for_inputs, text="Rozpocznij",
                            command=partial(startSimulation))
    startingButton.grid(row=7, column=1)

    # canvas_starting_photo.create_image(0, 0, anchor=NW, image=starting_photo)

    root.mainloop()

    # pydicom.config.image_handlers = [gdcm_handler, pillow_handler]
    # radon_transform("0002.dcm", 180, 1, 180)
    # reverse("0002.dcm")
