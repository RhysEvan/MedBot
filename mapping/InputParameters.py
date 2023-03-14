## Module for All Input Parameters of 3D scanner


## Input settings used in Camera1()
LeftCamera = "2BA200004266"                      ## Dependant on which usb portal is used
RightCamera = "2BA200004267"                     ## Dependant on which usb portal is used

## Parameters for generation of graycode patterns
width = 640
height = 480
image_resolution = (1536,864)

## Processing parameters used in Main()
binaryMaxValue = 1                  ## max value of the binary image. Should be 1 for processing. Should be 255 for viewing images.
#methodOfThreshold = 5                            ## 1=Simple thresholding,2=combined simple & mean,3=adaptive thresholding,4=mean per pixel thresholding,5=invers patterns,6=RobustPixelClassification
numberOfImages = 1                  ## number of higher frequency patterns that are not used if resolution proj. is higher than camera's. Limited resolution can mean less patterns results in higher resolution
methodOfTriangulation = 2

## Directories
ImageDirectory = r"C:\Users\mheva\OneDrive\Documents\GitHub\Stitching_Arm_Master_Thesis\mapping\Images"
WorkingDirectory = r"C:\Users\mheva\OneDrive\Documents\GitHub\Stitching_Arm_Master_Thesis\mapping"
CalibratedImageDirectory = r"C:\Users\mheva\OneDrive\Documents\GitHub\Stitching_Arm_Master_Thesis\mapping\CalibratedImages"
ChessboardImagesDirectory = r"C:\Users\mheva\OneDrive\Documents\GitHub\Stitching_Arm_Master_Thesis\mapping\ChessboardImages"
StereovisionDirectory = r"C:\Users\mheva\OneDrive\Documents\GitHub\Stitching_Arm_Master_Thesis\mapping\Stereovision"
imagesLeft = r"C:\Users\mheva\OneDrive\Documents\GitHub\Stitching_Arm_Master_Thesis\mapping\imagesLeft"
imagesRight = r"C:\Users\mheva\OneDrive\Documents\GitHub\Stitching_Arm_Master_Thesis\mapping\timagesRight"

