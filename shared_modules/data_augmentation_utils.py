from PIL import Image
import numpy as np

import imgaug.augmenters as iaa


class IdentityTransform:
    """A transform which does nothing (placeholder)
    
    Usually torchvision transform is subclass of torch.nn.Module, 
    it should implement the forward(self, img) method which 
    returns a img.
    
    Here we omit this for simplicity, this class does not have 
    a superclass. In consequence, instead of implementing 
    forward(self, img), we implement __call__(self, img) 
    which returns a img
    """

    def __init__(self):
        pass

    def __call__(self, img, **kwargs):
        """
        Args:
            Normally PIL Image or Tensor, 
            but here it can be anything
        """
        return img


class TransformBase:
    """
    order=0 (Nearest-neighbor)
    order=2 (Bi-quadratic)
    order=1 (Bi-linear)
    order=3 (Bi-cubic)
    order=4 (Bi-quartic)
    order=5 (Bi-quintic)
    """
    order = 3
    
    def __init__(self, random_seed=None):
        if random_seed is not None:
            random_seed = random_seed * 4 + 3
        self.rng = np.random.default_rng(random_seed)

    def __call__(self, img, prob=1., **kwargs):
        if self.rng.uniform(0, 1) > prob:
            return img
        img = self.forward(img)
        return img

    def forward(self, img, **kwargs):
        raise NotImplementedError


class RotateTransform(TransformBase):
    def __init__(self, degree=30, random_seed=None):
        super().__init__(random_seed)
        self.degree = degree

    def forward(self, img, **kwargs):
        """
        img
            PIL Image
        returns a PIL Image
        """
        img = np.asarray(img)
        iaa_transform = iaa.Rotate(
            (- self.degree, self.degree), order=self.order, mode="edge")
        img = iaa_transform(image=img)
        img = Image.fromarray(img)
        return img


class TranslateXTransform(TransformBase):
    def __init__(self, percent=0.1, random_seed=None):
        super().__init__(random_seed)
        self.percent = percent

    def forward(self, img, **kwargs):
        """
        img
            PIL Image
        returns a PIL Image
        """
        img = np.asarray(img)
        iaa_transform = iaa.TranslateX(
            percent=(- self.percent, self.percent), order=self.order, mode="edge")
        img = iaa_transform(image=img)
        img = Image.fromarray(img)
        return img


class TranslateYTransform(TransformBase):
    def __init__(self, percent=0.1, random_seed=None):
        super().__init__(random_seed)
        self.percent = percent

    def forward(self, img, **kwargs):
        """
        img
            PIL Image
        returns a PIL Image
        """
        img = np.asarray(img)
        iaa_transform = iaa.TranslateY(
            percent=(- self.percent, self.percent), order=self.order, mode="edge")
        img = iaa_transform(image=img)
        img = Image.fromarray(img)
        return img


class ShearXTransform(TransformBase):
    def __init__(self, degree=10, random_seed=None):
        super().__init__(random_seed)
        self.degree = degree

    def forward(self, img, **kwargs):
        """
        img
            PIL Image
        returns a PIL Image
        """
        img = np.asarray(img)
        iaa_transform = iaa.ShearX(
            shear=(- self.degree, self.degree), order=self.order, mode="edge")
        img = iaa_transform(image=img)
        img = Image.fromarray(img)
        return img


class ShearYTransform(TransformBase):
    def __init__(self, degree=10, random_seed=None):
        super().__init__(random_seed)
        self.degree = degree

    def forward(self, img, **kwargs):
        """
        img
            PIL Image
        returns a PIL Image
        """
        img = np.asarray(img)
        iaa_transform = iaa.ShearY(
            shear=(- self.degree, self.degree), order=self.order, mode="edge")
        img = iaa_transform(image=img)
        img = Image.fromarray(img)
        return img


class RandAugmentIrttssStraugV1:
    """
    In this implementation, the parameter M in RandAugment 
    is not used, the range of transform strength is either 
    fixed or randomly chosen among all possible values.
    
    The parameter prob is not used in the original RandAugment, 
    prob < 1 means that the untransformed images are more often 
    used. 
    
    N (int) follows the same notation as in the paper, it is the 
    number of transforms to apply sequentially to each image. 
    
    Irttss means: identity, rotation, translate-x, translate-y, 
        shear-x, shear-y
    """
    
    irttss_transforms = [
        IdentityTransform(), RotateTransform(degree=30),
        TranslateXTransform(percent=0.1),
        TranslateYTransform(percent=0.1),
        ShearXTransform(degree=10),
        ShearYTransform(degree=10)]

    def __init__(self, N=2, prob=1.,
                 random_seed=None, subset="IrttssStraug"):
        super().__init__()
        
        self.subset = subset
        
        if self.subset == "IrttssStraug":
            try:
                #from straug import warp
                #from straug import geometry
                #from straug import pattern
                from straug import blur
                from straug import noise
                from straug import weather
                from straug import camera
                from straug import process
                
                self.straug_transforms = [blur.DefocusBlur(), blur.MotionBlur(),
                                          blur.ZoomBlur(), 
                                          noise.GaussianNoise(), weather.Fog(),
                                          weather.Frost(), weather.Shadow(),
                                          camera.Contrast(), camera.Brightness(),
                                          camera.JpegCompression(), process.Posterize(),
                                          process.Solarize(), process.Invert(),
                                          process.Equalize(), process.AutoContrast(),
                                          process.Sharpness(), process.Color()]
            except Exception as e:
                print("""{}; straug or its dependency is not correctly installed:
                    https://github.com/roatienza/straug#pip-install""".format(e))
                raise e
            
            self.transforms = self.irttss_transforms + self.straug_transforms
        elif self.subset == "Irttss":
            self.transforms = self.irttss_transforms
        else:
            raise Exception("Uknown subset={}".format(self.subset))
        
        # N in the paper
        # number of transforms to apply sequentially to each image
        self.N = N
        self.prob = prob

        if random_seed is not None:
            random_seed = random_seed * 6 + 1
        self.rng = np.random.default_rng(random_seed)

    def __call__(self, img):
        sampled_transforms = self.rng.choice(self.transforms,
                                             size=self.N,
                                             replace=True)
        for transform in sampled_transforms:
            img = transform(img, prob=self.prob)
        return img
