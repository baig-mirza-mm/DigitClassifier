import pygame
from PIL import Image
import classify

# initialize
pygame.init()

window_height = 720
window_width = 1280

# create a 1280 x 720 display
screen = pygame.display.set_mode((1280,720))

clock = pygame.time.Clock()

running = True

mouse_pos = (0,0)
prev_mouse_pos = (0,0)

brush_radius = 18

probs_string = ""

# draw a circle at the start and end, and a line connecting the two points
def drawbrush(mouse_pos, prev_mouse_pos):
        pygame.draw.circle(screen, (255, 255, 255), mouse_pos, brush_radius * 0.9, 0)
        pygame.draw.line(screen, (255, 255, 255), mouse_pos, prev_mouse_pos, 2 * brush_radius)
        pygame.draw.circle(screen, (255, 255, 255), prev_mouse_pos, brush_radius * 0.9, 0)

# the drawing area is pure black
screen.fill((0, 0, 0))

font_large = pygame.font.SysFont("Consolas", 55, True)
font_small = pygame.font.SysFont("Arial", 40, True)

l_mouse_held = False

classification_ready = False

bounding_box_top_left = (720,720)
bounding_box_bottom_right = (0,0)

def screen_is_clear():
    return bounding_box_top_left == (720, 720) and bounding_box_bottom_right == (0,0)

def get_img_bounding_box():
    return (bounding_box_top_left, bounding_box_bottom_right)

# returns the pixels that were drawn
def get_drawing():
    output_image = Image.new("RGB", (bounding_box_bottom_right[0] - bounding_box_top_left[0], bounding_box_bottom_right[1] - bounding_box_top_left[1]), color = "black")

    for x in range(bounding_box_top_left[0], bounding_box_bottom_right[0]):
        for y in range (bounding_box_top_left[1], bounding_box_bottom_right[1]):
            # write all the RGB (discarding A) values
            output_image.putpixel((x - bounding_box_top_left[0], y - bounding_box_top_left[1]), screen.get_at((x, y))[:3])

    return output_image

def get_center_of_mass():
    weight_totals = 0
    contribution_totals = (0,0)

    for x in range(bounding_box_top_left[0], bounding_box_bottom_right[0]):
        for y in range(bounding_box_top_left[1], bounding_box_bottom_right[1]):
            pixel_color = screen.get_at((x,y))
            brightness = (pixel_color[0] + pixel_color[1] + pixel_color[2]) / (255 * 3)

            # to the totals, add each pixel weighted by its brightness
            contribution_totals = (contribution_totals[0] + x * brightness, contribution_totals[1] + y * brightness)

            weight_totals += brightness

    # The center of mass is the contribution totals divided by the weight totals
    return (contribution_totals[0] / weight_totals, contribution_totals[1] / weight_totals)

cycles = 0
while True:
    cycles = (cycles + 1) % 1000

    if (classification_ready and not screen_is_clear() and cycles % 5 == 0):
        probs_string = classify.classify(get_drawing())

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_g:
                print("Guessing...")
                probs_string = classify.classify(get_drawing())
                print(probs_string)

            # saves a preview of the parsed image going to the neural network
            if event.key == pygame.K_p:
                print(f"Saving preview... ({bounding_box_bottom_right[0] - bounding_box_top_left[0]}x{bounding_box_bottom_right[1] - bounding_box_top_left[1]})")
                classify.preview_image(get_drawing())
                print("Saved.")

            # debug for bounding box of drawing
            if event.key == pygame.K_b:
                print(f"Bounding Box\n\tTop Left:{bounding_box_top_left}\n\tBottom Right:{bounding_box_bottom_right}")
                pygame.draw.rect(screen, color=(0,0,255), rect = pygame.Rect(
                    bounding_box_top_left[0],
                    bounding_box_top_left[1],
                    bounding_box_bottom_right[0] - bounding_box_top_left[0],
                    bounding_box_bottom_right[1] - bounding_box_top_left[1]
                    )
                , width = 1)

            # debug for center of mass of drawing
            if event.key == pygame.K_c:
                center_of_mass = get_center_of_mass()
                print(f"Center of mass: {center_of_mass}")
                pygame.draw.circle(screen, (255,0,0), center_of_mass, 5)


        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                l_mouse_held = True

            xpos, ypos = event.pos

            # if the clear button is pressed then clear the screen and reset the bounding boxes
            if ((725 < xpos and xpos < 1275) and (670 < ypos and ypos < 715)):
                screen.fill((0, 0, 0))
                bounding_box_top_left = (720,720)
                bounding_box_bottom_right = (0,0)

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                l_mouse_held = False

        if event.type == pygame.MOUSEMOTION:
            xpos, ypos = event.pos
            mouse_pos = (xpos,ypos)

    if (l_mouse_held and mouse_pos[0] <= 720 and mouse_pos[1] <= 720):
        # set the bounding box coords if the position of a new pixel is out of the current bounds
        bounding_box_tl_x = max(min(mouse_pos[0] - brush_radius, bounding_box_top_left[0]), 0)
        bounding_box_tl_y = max(min(mouse_pos[1] - brush_radius, bounding_box_top_left[1]), 0)
        bounding_box_br_x = min(max(mouse_pos[0] + brush_radius, bounding_box_bottom_right[0]), 720)
        bounding_box_br_y = min(max(mouse_pos[1] + brush_radius, bounding_box_bottom_right[1]), 720)

        bounding_box_top_left = (bounding_box_tl_x, bounding_box_tl_y)
        bounding_box_bottom_right = (bounding_box_br_x, bounding_box_br_y)

        drawbrush(mouse_pos, prev_mouse_pos)

    prev_mouse_pos = mouse_pos

    # specify the non-drawable area
    info_box = pygame.draw.rect(screen, (10,10,10), (720, 0, 1280 - 720, 720))

    # the probabilities that goes into the non-drawable area
    probs_text = font_large.render(probs_string, True, (255, 255, 255))

    # the "clear" button
    clear_rect = pygame.draw.rect(screen, (30, 30, 30), (720 + 5, 720 - 50, 560 - 5 - 5, 45))

    clear_text = font_small.render("Clear", True, (255, 255, 255))

    # change the clear rect so that the text is centered
    clear_rect = clear_text.get_rect(center=(720 + (1280 - 720) / 2, 720 - 25))

    # display the text
    screen.blit(clear_text, clear_rect)
    screen.blit(probs_text, info_box)

    pygame.display.flip()

    clock.tick(120)

    # ready to begin classifying numbers
    classification_ready = True
