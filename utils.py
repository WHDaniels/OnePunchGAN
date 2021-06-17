from torchvision import transforms as ts


def get_augmented_images(image_A, image_B, mode):
    # Apply transformations:
    # horizontal flips --> 2x
    # rotating (non 90, 180, 270 degree) --> 4x
    # cropping (four quadrants) --> 4x
    # 2*4*4 = 32x original dataset

    if mode == 'train':

        augmentation = ts.Compose([
            ts.RandomHorizontalFlip(p=0.05),
            ts.RandomRotation((0, 360), fill=255),
            ts.Resize((image_A.size, image_A.size), interpolation=ts.InterpolationMode.BICUBIC),
            ts.ToTensor(),
            ts.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        return {'A': augmentation(image_A),
                'B': augmentation(image_B)}

    else:

        augmentation = ts.Compose([
            # ts.RandomHorizontalFlip(p=0.05),
            # ts.RandomRotation((0, 360), fill=255),
            ts.Resize((image_B.size, image_B.size), interpolation=ts.InterpolationMode.BICUBIC),
            ts.ToTensor(),
            ts.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        return {'A': augmentation(image_A),
                'B': augmentation(image_B)}
