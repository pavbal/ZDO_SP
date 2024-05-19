import sys
import numpy as np
from scipy import ndimage
from skimage import exposure
import skimage.feature
import skimage.io
from skimage import io, measure
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def horizontalize(img_grayscale):
    img_gs = 1 - img_grayscale
    img_gs = exposure.rescale_intensity(img_gs, in_range='image', out_range=(0.01, 1))

    c = np.shape(img_gs)[1] // 5
    img_cut_3 = img_gs[:, c:-c]
    degrees = range(-10, 10, 1)
    maximums = np.zeros(len(degrees))

    for i in range(0, len(degrees)):
        imr = skimage.transform.rotate(img_cut_3, degrees[i], resize=True)
        sum_hist = np.sum(imr, axis=1)
        maximums[i] = np.max(sum_hist)

    uhel = degrees[np.argmax(maximums)]
    img_str = skimage.transform.rotate(img_gs, uhel, resize=False)
    if np.argmax(maximums) < len(degrees) // 2:
        sign = -1
    else:
        sign = 1


    mask = (img_str != 0.0)
    argm0 = np.argmax(mask[::sign, 0])
    argm1 = np.argmax(mask[0, ::-sign])

    img_cropped = img_str[argm0 + 1:-(argm0 + 1), argm1 + 1:-(argm1 + 1)]
    img_cropped = exposure.rescale_intensity(img_cropped, in_range='image', out_range=(0, 1))

    return img_cropped, uhel, [argm0, argm1]


def cut_sides(img_gs):
    img_sobel = abs(ndimage.sobel(img_gs, axis=0))

    mask = np.zeros(np.shape(img_sobel))
    num = len(mask[:, 0]) // 7
    mask[num:-num, :] = 1

    img_bi = img_sobel > 1.8 * np.mean(img_sobel)  # 0.5
    img_bi = img_bi * mask
    kernel_1 = np.ones(len(img_gs[0, :]) // 14, dtype=np.uint8)

    kernel_0 = np.array([[1], [1], [1]], dtype=np.uint8)
    kernel_1 = np.array([kernel_1], dtype=np.uint8)
    kernel = skimage.morphology.diamond(1)
    kernel_2 = skimage.morphology.rectangle(len(img_gs[:, 0]) // 10, len(img_gs[0, :]) // 20)

    img_bin = ndimage.binary_dilation(img_bi, kernel_0, iterations=1)

    img_bin = ndimage.binary_fill_holes(img_bin)

    # img_bin = ndimage.binary_dilation(img_fill, kernel_0, iterations=1)
    img_bin = ndimage.binary_opening(img_bin, kernel_1, iterations=1)
    img_bin = ndimage.binary_closing(img_bin, kernel, iterations=1)
    # img_bin = ndimage.binary_erosion(img_bin, kernel_2, iterations=1)
    img_bin = ndimage.binary_closing(img_bin, kernel_1, iterations=1)
    img_bin = ndimage.binary_erosion(img_bin, kernel_0, iterations=1)
    img_bin = ndimage.binary_fill_holes(img_bin)

    vect = np.sum(img_bin, axis=0)
    arg1 = np.argmax(vect > 0)
    arg2 = len(vect) - np.argmax(vect[::-1] > 0)

    vect2 = np.sum(img_bin, axis=1)
    arg1_1 = np.argmax(vect2 > 0)
    arg2_1 = len(vect2) - np.argmax(vect2[::-1] > 0)

    img_sidecut = img_gs[:,arg1:arg2]
    return img_sidecut, img_bin, [arg1, arg2], [arg1_1, arg2_1]




def segmentate_1D(img_gs, procento=87):
    img_sobel = abs(ndimage.sobel(img_gs, axis=1))

    sorted_pixels = np.sort(img_sobel.flat)
    pixels_count = len(sorted_pixels)
    threshold_index = int(pixels_count * 0.01 * procento)
    thresh = sorted_pixels[threshold_index]

    img_binary = img_sobel > thresh

    # kernels
    kernel_1 = np.ones(len(img_gs[:, 0]) // 8, dtype=np.uint8)
    kernel_3 = np.ones(len(img_gs[0, :]) // 45, dtype=np.uint8)
    kernel_2 = np.ones(len(img_gs[0, :]) // 35, dtype=np.uint8)
    kernel_4 = np.ones(len(img_gs[0, :]) // 50, dtype=np.uint8)

    kernel_1 = np.transpose(np.array([kernel_1], dtype=np.uint8))
    kernel_2 = np.array([kernel_2], dtype=np.uint8)
    kernel_3 = np.array([kernel_3], dtype=np.uint8)
    kernel_4 = np.array([kernel_4], dtype=np.uint8)
    kernel_5 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    try:
        img_binary = ndimage.binary_opening(img_binary, kernel_4, iterations=1)
        img_binary = ndimage.binary_closing(img_binary, kernel_2, iterations=1)
        img_binary = ndimage.binary_fill_holes(img_binary)

        img_binary = ndimage.binary_opening(img_binary, kernel_3, iterations=1)
        img_binary = ndimage.binary_opening(img_binary, kernel_2, iterations=1)
        img_binary = ndimage.binary_opening(img_binary, kernel_1, iterations=1)
        img_binary = ndimage.binary_opening(img_binary, kernel_5, iterations=1)

        # pseudo 1D
        kernel_6 = np.ones(len(img_gs[:, 0]) * 2, dtype=np.uint8)
        kernel_6 = np.transpose(np.array([kernel_6], dtype=np.uint8))
        img_binary_count = ndimage.binary_dilation(img_binary, kernel_6, iterations=1)
    except:
        return None, None

    return img_binary_count, img_binary

def differentiate(gap_detect):
    num_gaps = np.max(gap_detect)

    diff = np.diff(gap_detect)
    if diff[np.argwhere(diff)[0]] == -1:
        if diff[0] == -1:
            diff[1] = -1
        diff[0] = 1
    if gap_detect[-1] == num_gaps:
        if diff[-1] == num_gaps:
            diff[-2] = num_gaps
        diff[-1] = -num_gaps
    return diff, num_gaps


def xml_to_stitches(doc, index):
    stitches = []
    if "polyline" in doc["annotations"]["image"][index]:
        # if "Stitch" in doc["annotations"]["image"][index]["polyline"]:
        #     continue
        try:
            for pline in doc["annotations"]["image"][index]["polyline"]:
                    if pline["@label"] == "Incision":
                        continue
                    pts = np.array([pt.split(",") for pt in pline["@points"].split(";")], dtype=float)
                    stitches.append(pts)
        except:
            print("image with index ", index, " is not usable")

    return stitches

def print_stitch_stats(width, height):
    print("Stitchables stats: ")
    print("")
    print("\tmaximum height:\t", np.max(height))
    print("\tmaximum width: \t", np.max(width))
    print("")
    print("\tminimum height:\t", np.min(height))
    print("\tminimum width:\t", np.min(width))
    print("")
    print("\tmedian height:\t", sorted(height)[len(height) // 2])
    print("\tmedian width:\t", sorted(width)[len(width) // 2])
    print("")
    print("\tmean height:\t", np.mean(height))
    print("\tmean width:  \t", np.mean(width))
    print("")
    print("\tmean ratio:  \t", np.mean(np.divide(height,width)))
    print("")
    print("\tsorted heights:\t", sorted(height))
    print("\tsorted widths: \t", sorted(width))

    counts1 = Counter(height)
    counts2 = Counter(width)
    # Rozdělíme data na hodnoty a jejich četnosti
    values1, frequencies1 = zip(*counts1.items())
    values2, frequencies2 = zip(*counts2.items())


    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.bar(values1, frequencies1)
    plt.xlabel('Výška obrázku')
    plt.ylabel('Četnost')
    plt.title('Rozložení výšky obrázků v datasetu steh-nesteh')
    plt.subplot(212)

    plt.bar(values2, frequencies2)
    plt.xlabel('Šířka obrázku')
    plt.ylabel('Četnost')
    plt.title('Rozložení šířky obrázků v datasetu steh-nesteh')

    plt.subplots_adjust(hspace=0.5)

    plt.show()


def preprocess_and_cut_fcn(img):

    img_gs = skimage.color.rgb2gray(img)

    img_gs_hor, uhel, argms = horizontalize(img_gs)
    img_gs_cut, img_bin, args, args2 = cut_sides(img_gs_hor)
    img_gs_cut = img_gs_cut[:,4:-4]
    img_sobel = abs(ndimage.sobel(img_gs_cut, axis=1))
    img_binary_count, img_binary = segmentate_1D(img_gs_cut, procento=82)

    if img_binary is not None:

        kernel = skimage.morphology.diamond(1)
        img_binary_count = ndimage.binary_dilation(img_binary_count, kernel, iterations=1)

        # tady bude anotace, bc == binary count
        img_bc_label = measure.label(img_binary_count)
        bc_label_1D = img_bc_label[0,:]
        num_objects = np.max(bc_label_1D)
        gap_detect = measure.label(1-img_binary_count)[0,:]

        diff, num_gaps = differentiate(gap_detect)

        border_pts = []

        if bc_label_1D[0]!=0:
            border_pts.append(0)
        for i in range(1, num_gaps+1):
            a = np.argwhere(np.absolute(diff) == i)[0]
            b = np.argwhere(np.absolute(diff) == i)[1]
            center = ((a+b)//2)[0]
            border_pts.append(center)
        if bc_label_1D[-1]!=0:
            border_pts.append(len(bc_label_1D)-1)

        stitch_images = []
        for i in range(0, num_objects):
            stitch_image = img_gs_cut[:,border_pts[i]:border_pts[i+1]]
            stitch_images.append(stitch_image)


        img_gs_fragm = (1-img_gs_cut)
        img_gs_fragm[:, border_pts] = 0
        img_gs_fragm[:, 0:border_pts[0]] = 0
        img_gs_fragm[:, border_pts[-1]:] = 0

        process = {
            "img_binary_count": img_binary_count,
            "img_binary": img_binary,
            "img_horizontalized": img_gs_hor,
            "img_cut_sides": img_gs_cut,
            "img_sobel": img_sobel,
            "img_gs": img_gs,
            "img_gs_fragm": img_gs_fragm,
            "border_pts": border_pts
        }

    return stitch_images, border_pts, process

def parse_arguments() -> tuple[str, bool, list[str]]:
    # @author Pavel Březina
    def print_help():
        print("Invalid arguments supplied.\nExample usage:\npython run.py output.csv incision001.jpg incision005.png incision010.JPEG")

    if len(sys.argv) < 3:
        print_help()
        exit(-1)

    csv_file = sys.argv[1]
    visual_mode = False
    input_files = []

    for i in range(2, len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-v":
            visual_mode = True
        else:
            input_files.append(arg)

    if len(input_files) == 0:
        print_help()
        exit(-1)

    return (csv_file, visual_mode, input_files)


def visualize(process, img, classes):
    fig = plt.figure(figsize=(4, 8))
    # plt.subplot(5, 1, 1)
    ax = fig.add_subplot(5, 1, 1)
    ax.set_title("původní obrázek")
    ax.imshow(img, cmap='gray')
    ax.axis('off')

    ax = fig.add_subplot(5, 1, 2)
    ax.set_title("narovnaný a oříznutý (šedotón, negativ)")
    ax.imshow(process["img_cut_sides"], cmap='gray')
    ax.axis('off')

    ax = fig.add_subplot(5, 1, 3)
    ax.set_title("segmentace (ne)stehů")
    ax.imshow(process["img_binary"], cmap='gray')
    ax.axis('off')

    ax = fig.add_subplot(5, 1, 4)
    ax.set_title("maskování (ne)stehů")
    ax.imshow(process["img_binary_count"] * (1 - process["img_cut_sides"]), cmap='gray')
    ax.axis('off')

    pts = process["border_pts"]

    ax = fig.add_subplot(5, 1, 5)
    ax.set_title("rozdělení na stehy a klasifikace")
    ax.imshow(1-process["img_cut_sides"], cmap='gray')

    for i in range(len(classes)):
        if classes[i]==1:
            edgecolor = 'g'
        else:
            edgecolor = 'r'
        rect = patches.Rectangle((pts[i], 0), pts[i+1] - pts[i]-1, len(process["img_cut_sides"][:,0])-1, linewidth=1, edgecolor=edgecolor, facecolor='none')
        ax.add_patch(rect)
    ax.axis('off')
    fig.suptitle('Vizualizace postupu a predikce', fontsize=16)

    plt.show()

    return
