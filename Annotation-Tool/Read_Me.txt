========================== Image Annotation tool ==========================
The image annotation tool is an image editting tool that allows the user to label certain features in the image.
This tool can be used to produce labeled training data for AI usage.
The images can be in H5 and jpg format.
For H5 files:  The labels are created on a seperate layer in the H5 file.
For jpg files: The labels are created on a new image (format: filename_label.jpg)

How does it work:
Upon running the script, the user is prompted with the selection menu. In here, a path to the unlabeled images* and a path to the label excell**
must be selected. When the user presses 'Done' the annotation window will open.

*   This folder contains the images that the user wants to label in H5 format.
**  The user can create their own labels in an excel file.

The Annotation window contains the following:

o  In the center the image that needs to be labeled
o  The different labels represented in the excell file on the left
o  Hotkey legend to the right
o  Current label and current filename at the top

Controls:

If the user clicks in the image, a point is placed. select multiple points to create an area. This area contains the feature that needs to be labeled.
Arrow up and down change the label. This means that the user can have multiple (up to 40) labels per image.
If multiple unconnected parts of one unique feature must be selected, the user can right-click to create a new seperate area with the same label name.
In case there are multiple layers in the image, the user can circle through them by pressing '1' to go back or '2' to go forward.
To switch between images, press the left key to go back and the right key to go forward.
An incorrectly placed point can be removed by pressing backspace.

We have opted to use only 10 different colors for different labels, as colors might look to much alike. In order to still have plenty of labels, a new set of
10 labels is recognisable by its unique point icon and its different color in-fill. The opacity of the fill can be changed by pressing 'f'.

Note:
Images that are labeled get '_labeled' added to their file name and will NOT be opened again by the annotation tool. If the user wishes to open a file
again to redo the labels, either manually change the file name or run the map through the included renamer file.

Do not spam the hotkeys. Loading large H5 images can take a while.

Credits:

Original annotation tool by Edgar Cardenas
Editted by Rhys Evans, Gilles Vanlommel, Brent Cleys, Yorne Van Praet