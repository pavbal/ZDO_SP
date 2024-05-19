import xmltodict
from src.functions import horizontalize, cut_sides, segmentate_1D
import numpy as np
from scipy import ndimage
from skimage import io, measure
import skimage.feature
import matplotlib.pyplot as plt
from preprocess_and_cut_fcn import preprocess_and_cut_fcn

# index = 75, 38

index = 38


with open('../annotations.xml') as fd:
    doc = xmltodict.parse(fd.read())


name = doc["annotations"]["image"][index]["@name"]
name1 = name[16:]
print(name1)
# file_list = os.listdir("incision_couples")
# file_list.sort()
# image_path = os.path.join("incision_couples", file_list[index])
# img= io.imread(image_path)
print("name: ", name)
img = io.imread(name)

img_gs = skimage.color.rgb2gray(img)




stitches = []
for pline in doc["annotations"]["image"][index]["polyline"]:
    # extract coodrinates
    if pline["@label"] == "Incision":
        continue
    pts = np.array([pt.split(",") for pt in pline["@points"].split(";")], dtype=float)
    stitches.append(pts)
    # plt.plot(pts[:, 0], pts[:, 1])







# záznam stehů
img_gs_stitch = np.zeros(np.shape(img_gs))
for stitch in stitches:
    img_gs_stitch[round(stitch[0,1]), round(stitch[0,0])] = 1
    img_gs_stitch[round(stitch[1,1]), round(stitch[1,0])] = 1


# img_gs = img[:,:,0]
img_gs, uhel, argms = horizontalize(img_gs)
img_gs_stitch = skimage.transform.rotate(img_gs_stitch, uhel, resize=False)
img_gs_stitch = img_gs_stitch[argms[0] + 1:-(argms[0] + 1), argms[1] + 1:-(argms[1] + 1)]

img_gs_rot = img_gs

img_gs, img_bin, args, args2 = cut_sides(img_gs)
img_gs_stitch = img_gs_stitch[:,args[0]:args[1]]

img_gs = img_gs[:,4:-4]
img_gs_stitch = img_gs_stitch[:,4:-4]

img_binary_count, img_binary = segmentate_1D(img_gs, procento=82)
# print(img_gs)
img_sobel = abs(ndimage.sobel(img_gs, axis=1))

kernel = skimage.morphology.diamond(1)
img_binary_count = ndimage.binary_dilation(img_binary_count, kernel, iterations=1)

# tady bude anotace, bc == binary count
img_bc_label = measure.label(img_binary_count)
bc_label_1D = img_bc_label[0,:]
# print(bc_label_1D)
indicator = (np.sum(img_gs_stitch, axis=0)>0)
labels = []
num_objects = np.max(bc_label_1D)
for i in range(0, num_objects):
    is_stitch = np.sum(np.multiply((bc_label_1D==i+1), indicator))>0
    labels.append(is_stitch)
# print(labels)

gap_detect = measure.label(1-img_binary_count)[0,:]
num_gaps = np.max(gap_detect)
# num_stitchables = np.max(bc_label_1D)

# print("gap_detect")
# print(gap_detect)
# plt.imshow(img_binary_count, cmap='gray')
# plt.show()

diff = np.diff(gap_detect)
print(diff)
if diff[np.argwhere(diff)[0]] == -1:
    if diff[0] == -1:
        diff[1] = -1
    diff[0] = 1
if gap_detect[-1] == num_gaps:

    diff[-1] = -num_gaps
print(diff)

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
## ukládání
# for i in range(0, num_objects):
#     # print(np.shape(img_gs), "  -  ", border_pts[i], "   for  i == ", i)
#     stitch_image = img_gs[:,border_pts[i]:border_pts[i+1]]
#     stitch_images.append(stitch_image)
#     plt.imsave('stitches/'+name1[:-4]+'__'+ str(i) + '_'+ str(int(labels[i])) +'.png', stitch_image, cmap='gray')

# plt.figure(figsize=(13,9))
# plt.subplot(5,2,1)
# plt.title("původní obrázek")
# plt.imshow(img, cmap='gray')
# plt.subplot(522)
# plt.title("hrubá segmentace narovnané rány pro určení oříznutí ze stran")
# plt.imshow(img_bin, cmap='gray')
# plt.subplot(523)
# plt.title("narovnaný obrázek podle roviny rány (incision)")
# plt.imshow(1-img_gs, cmap='gray')
# plt.subplot(524)
# plt.title('stitch')
# plt.imshow(img_gs_stitch, cmap='gray')
# plt.subplot(525)
# plt.title("sobel - vertikální hrany")
# plt.imshow(img_sobel, cmap='gray')
# plt.subplot(526)
# plt.title("po sofis. prahování abs vert. sobela a po morf. úpravách")
# plt.imshow(img_binary, cmap='gray')
# plt.subplot(527)
# plt.title("")
# plt.imshow(img_gs*img_binary_count, cmap='gray')
# plt.subplot(528)
# plt.title("Možná maska pro budoucí zpracování (extr. přízn.) a klasifikaci")
# plt.imshow(img_binary_count, cmap='gray')
# plt.subplot(529)
# plt.title("")
# plt.imshow(img_bc_label, cmap='gray')


# fig = plt.figure(figsize=(8, 6))
# plt.subplot(3,2,1)
# plt.title("původní obrázek rány")
# plt.imshow(img, cmap='gray')
# plt.subplot(323)
# plt.title("šedotónový obrázek původní rány")
# plt.imshow(skimage.color.rgb2gray(img), cmap='gray')
# plt.subplot(325)
# plt.title("narovnání dle roviny rány")
# plt.imshow(1-img_gs_rot, cmap='gray')
# plt.subplot(322)
# plt.title("absolutní výstup Sobelova filtru (horiz.)")
# plt.imshow(img_sobel, cmap='gray')
# plt.subplot(324)
# plt.title("hrubá segm. narovnané rány")
# plt.imshow(img_bin, cmap='gray')
# plt.subplot(326)
# plt.title("oříznutí na základě segm. rány")
# plt.imshow(1-img_gs, cmap='gray')
# # plt.tight_layout()
# fig.suptitle('Předzpracování - horizontalizace a oříznutí, příklad', fontsize=16) # or plt.suptitle('Main title')
# plt.show()

stitch_imgs, pts, _ = preprocess_and_cut_fcn(img)
img_gs_fragm = (1-img_gs)
img_gs_fragm[:, pts] = 0
img_gs_fragm[:, 0:pts[0]] = 0
img_gs_fragm[:, pts[-1]:] = 0

fig = plt.figure(figsize=(9, 6))
plt.subplot(3,2,1)
plt.title("výstup předchozího zpracování")
plt.imshow(1-img_gs, cmap='gray')
plt.subplot(323)
plt.title("absolutní výstup Sobelova filtru (vert.)")
plt.imshow(img_sobel, cmap='gray')
plt.subplot(325)
plt.title("segmentace stehů a nestehů")
plt.imshow(img_binary, cmap='gray')
plt.subplot(322)
plt.title("převedení na 1D segmentaci")
plt.imshow(img_binary_count, cmap='gray')
plt.subplot(324)
plt.title("vymaskování stehů")
plt.imshow(img_binary_count*(1-img_gs), cmap='gray')
plt.subplot(326)
plt.title("rozdělení na obrázky stehů")
plt.imshow(img_gs_fragm, cmap='gray')
# plt.tight_layout()
fig.suptitle('Předzpracování - detekce stehů a nestehů, příklad', fontsize=16) # or plt.suptitle('Main title')
plt.show()