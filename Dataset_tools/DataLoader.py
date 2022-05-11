from torchvision.datasets import ImageFolder
class ImageFolderWithPaths(ImageFolder):
    """
        Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder

        Facilita para verificar as imagens onde o modelo esta errando/acertando,
    caso desejado
    """
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        #path = self.imgs[index][0]
        #tuple_with_path = (original_tuple + (path,))

        return original_tuple