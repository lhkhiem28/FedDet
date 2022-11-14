
import os, sys
from libs import *

class DetImageDataset(torch.utils.data.Dataset):
    def __init__(self, 
        images_path, labels_path
        , image_size = 416
        , augment = False
        , multiscale = False
    ):
        self.image_files, self.label_files = sorted(glob.glob(images_path + "/*")), sorted(glob.glob(labels_path + "/*"))
        self.image_size = image_size
        self.augment = augment
        self.transforms = A.Compose(
            [
                A.HorizontalFlip(
                    p = 0.5, 
                ), 
                A.BBoxSafeRandomCrop(
                    erosion_rate = 0.2, 
                    p = 0.5, 
                ), 
                A.RandomBrightnessContrast(
                    brightness_limit = 0.2, contrast_limit = 0.2, 
                    p = 0.3, 
                ), 
                A.RGBShift(
                    r_shift_limit = 30, g_shift_limit = 30, b_shift_limit = 30, 
                    p = 0.3, 
                ), 
            ], 
            A.BboxParams("yolo", ["classes"])
        )
        self.multiscale = multiscale
        self.image_sizes = [self.image_size + 32*scale for scale in range(-1, 2)]

    def __len__(self, 
    ):
        return len(self.image_files)

    def square_pad(self, 
        image, 
    ):
        _, h, w = image.shape
        gap_pad = np.abs(h - w)
        if h - w < 0:
            pad = (0, 0, gap_pad // 2, gap_pad - gap_pad // 2)
        else:
            pad = (gap_pad // 2, gap_pad - gap_pad // 2, 0, 0)

        image = F.pad(
            image, 
            pad = pad, value = 0.0, 
        )
        return image, [0] + list(pad)

    def __getitem__(self, 
        index, 
    ):
        image_file, label_file = self.image_files[index], self.label_files[index]
        image = cv2.imread(image_file)
        image = cv2.cvtColor(
            image, 
            code = cv2.COLOR_BGR2RGB, 
        )
        bboxes = np.loadtxt(label_file)
        bboxes = bboxes.reshape(-1, 5)
        if self.augment:
            Transformed = self.transforms(
                image = image, 
                classes = bboxes[:, 0], bboxes = bboxes[:, 1:]
            )
            image = Transformed["image"]
            bboxes[:, 1:] = np.array(Transformed["bboxes"])

        image = torch.tensor(image)
        image = image.permute(2, 0, 1)
        _, h, w = image.shape
        image, pad = self.square_pad(image); _, padded_h, padded_w = image.shape
        c1, c2, c3, c4,  = w*(bboxes[:, 1] - bboxes[:, 3]/2) + pad[1], w*(bboxes[:, 1] + bboxes[:, 3]/2) + pad[2], h*(bboxes[:, 2] - bboxes[:, 4]/2) + pad[3], h*(bboxes[:, 2] + bboxes[:, 4]/2) + pad[4], 
        bboxes[:, 1], bboxes[:, 2], bboxes[:, 3], bboxes[:, 4],  = ((c1 + c2)/2)/padded_w, ((c3 + c4)/2)/padded_h, bboxes[:, 3]*(w/padded_w), bboxes[:, 4]*(h/padded_h), 
        return image.float(), F.pad(
            torch.tensor(bboxes), 
            pad = (1, 0, 0, 0), value = 0.0, 
        )

    def collate_fn(self, 
        batch, 
    ):
        images, labels = list(zip(*batch))
        if self.multiscale and np.random.random() <= 0.1:
            self.image_size = np.random.choice(self.image_sizes)
        images = torch.stack([
            F.interpolate(
                image.unsqueeze(0), 
                self.image_size, mode = "nearest", 
            ).squeeze(0) for image in images
        ])
        images = images/255

        labels = [bboxes for bboxes in labels if bboxes is not None]
        for index, bboxes in enumerate(labels):
            bboxes[:, 0] = index
        if len(labels) != 0:labels = torch.cat(labels)
        return images, labels