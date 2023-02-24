## Module for All Input Parameters of 3D scanner


## Input settings used in Camera1()
LeftCamera = 0                      ## Dependant on which usb portal is used
RightCamera = 4                     ## Dependant on which usb portal is used

## Parameters for generation of graycode patterns
width = 640
height = 480
image_resolution = (1536,864)

## Processing parameters used in Main()
binaryMaxValue = 1                  ## max value of the binary image. Should be 1 for processing. Should be 255 for viewing images.
methodOfThreshold = 5                            ## 1=Simple thresholding,2=combined simple & mean,3=adaptive thresholding,4=mean per pixel thresholding,5=invers patterns,6=RobustPixelClassification
numberOfImages = 1                  ## number of higher frequency patterns that are not used if resolution proj. is higher than camera's. Limited resolution can mean less patterns results in higher resolution
methodOfTriangulation = 2

## Directories
ImageDirectory = r"/home/yuno/anaconda3/envs/camera/Masterproef_JaccoFrée_2022/pythonProject/Images"
WorkingDirectory = r"/home/yuno/anaconda3/envs/camera/Masterproef_JaccoFrée_2022/pythonProject"
CalibratedImageDirectory = r"/home/yuno/anaconda3/envs/camera/Masterproef_JaccoFrée_2022/pythonProject/CalibratedImages"
ChessboardImagesDirectory = r"/home/yuno/anaconda3/envs/camera/Masterproef_JaccoFrée_2022/pythonProject/ChessboardImages"
StereovisionDirectory = r"/home/yuno/anaconda3/envs/camera/Masterproef_JaccoFrée_2022/pythonProject/Stereovision"
imagesLeft = r"/home/yuno/anaconda3/envs/camera/Masterproef_JaccoFrée_2022/pythonProject/imagesLeft"
imagesRight = r"/home/yuno/anaconda3/envs/camera/Masterproef_JaccoFrée_2022/pythonProject/imagesRight"

