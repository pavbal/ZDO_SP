import csv

from functions import parse_arguments, preprocess_and_cut_fcn, visualize
from nns.Net import SimpleCNN, predict, load_model_SimpleCNN
from skimage import io



if __name__ == "__main__":
    csv_file, visual_mode, image_paths = parse_arguments()

    ## možné spuštění
    # csv_file = "output.csv"
    # visual_mode = True
    # image_paths = ["../incision_couples/SA_20230222-130319_c1g5pmjim3m6_incision_crop_0.jpg", "../incision_couples/SA_20220620-102621_8ka1kmwpywxv_incision_crop_0.jpg"]

    image_paths = list(image_paths)
    model = load_model_SimpleCNN("../models/model_all_in_800.pth")

    images = []
    for path in image_paths:
        images.append(io.imread(path))

    n_stitches = []
    for img in images:
        stitch_images, border_pts, process = preprocess_and_cut_fcn(img) # preprocessing a rozsekání na stehy
        num_stitches = 0
        stitchable_class = []
        for stitchable in stitch_images:
            prediction = predict(model, stitchable, 1)
            num_stitches += prediction
            stitchable_class.append(prediction)

        n_stitches.append(num_stitches)
        if visual_mode:
            visualize(process, img, stitchable_class)

    data = []
    for i in range(len(n_stitches)):
        data.append({'filename': image_paths[i], 'n_stitches': n_stitches[i]})

    # Zápis do CSV souboru
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['filename', 'n_stitches'])

        # Zapsání hlavičky
        writer.writeheader()

        # Zapsání dat
        for row in data:
            writer.writerow(row)

    print(f'Data byla úspěšně uložena do souboru {csv_file}')

