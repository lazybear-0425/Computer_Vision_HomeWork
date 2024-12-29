import pygame
import sys

pygame.init()
screen = pygame.display.set_mode((700,400))

plus_ball = pygame.image.load('term_project/images/minus_ball.png')
plus_ball = pygame.transform.scale(plus_ball, (25, 25))

while True:
    screen.fill((255, 255, 255))
    rect = pygame.draw.rect(screen, (255, 0, 0), (100, 100, 10, 10))
    screen.blit(plus_ball, rect)
    for event in pygame.event.get(): #回傳list of Event Object
        if event.type == pygame.QUIT: #按下右上角的X
            pygame.quit()
            sys.exit()
    pygame.display.update() #畫完才一次更新