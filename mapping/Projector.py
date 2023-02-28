import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"           ## Parameter to turn off Pygame console prints
import pygame
from .InputParameters import *

def imgToScrn(scrn,imgFilename):                ## Function to control projector and display full screen
    pygame.init()
    monitor_size = [pygame.display.Info().current_w, pygame.display.Info().current_h]
    screen = pygame.display.set_mode(monitor_size, pygame.FULLSCREEN)

    os.chdir(InputParameters.ImageDirectory)
    img = pygame.image.load(imgFilename)
    siRect = img.get_rect()
    screen.blit(img,siRect)
    pygame.display.flip()

def DestroyWindow():
    pygame.display.quit()



