#### Master Thesis: Autonomous robot used for medical applications

Overview

This project presents a robotic arm mounted onto a large XY-translation stage, designed to perform precise movements and tasks.
The system integrates multiple submodules to enhance functionality and user interaction. A graphical user interface (GUI) is included to simplify user control and operation.
The original purpose was to detect lacerations and to autonomously stitch by projecting the coordinates onto a 3D scan of the patient.
The project was done in less than a year with a group of six students.
2 Master students and 4 bachelor students.
I was lead organiser and sole programmer whilst the other group members focused on the mechanical side.

## Submodules

# Arduino_robot_control
  Facilitates interaction between the robotic arm and an Arduino microcontroller.
  Ensures efficient command transmission.
# Inverse Kinematics Machine Learning Module
  Implements machine learning methods to solve inverse kinematics problems.
  Enhances movement precision and adaptability for various tasks.
# Triangulation Module for 3D Scanning
  Utilizes structured light (gray-code) to perform 3D scanning.
  Enables object recognition and environmental mapping.
# Retna
  Tool for semantic segmentation for laceration detection.


## Usage

Launch the GUI and select the desired operational mode.
Input movement commands or use the inverse kinematics module for automated positioning.
Utilize the 3D scanning feature for structured light triangulation.

## Restuls

The performance of the laceration detection was reliable and robust on cadavers and synthetic skin samples:
  ![image](https://github.com/user-attachments/assets/23193109-ef9f-43e1-91ce-e7dcb9d904ba)

The prediction quality of the Inverse Kinematics neural network resulted in equal quality predictions to that of an empirical method such as a Jacobian or Heuristic method when the robot had 8 DOF.
  ![image](https://github.com/user-attachments/assets/ffddbd20-145f-4fbb-9a4d-4b7047b7aa51)

Gui was programmed in PyQt5 and was user friendly, it was tested on other students following the engineering course who quickly knew how to control it.
  ![image](https://github.com/user-attachments/assets/f5c98158-81ef-4c84-b909-028f39b72117)


## Future Work

Enhancing machine learning algorithms for better kinematics predictions.
Expanding the structured light scanning capabilities and integrating it into gui.
Implementing additional automation features such as path planning to feature of interest.
