from PIL import Image
import torch
from torchvision import transforms
from model import Network

model = Network()
model.load_state_dict(torch.load("mnist.pth", map_location="cpu"))
model.eval()

def get_center_of_mass(img: Image.Image):
    width, height = img.size

    weight_totals = 0
    contribution_totals = (0,0)

    for x in range(width):
        for y in range(height):
            pixel_color = img.getpixel((x,y))
            brightness = (pixel_color[0] + pixel_color[1] + pixel_color[2]) / (255 * 3)

            # to the totals, add each pixel weighted by its brightness
            contribution_totals = (contribution_totals[0] + x * brightness, contribution_totals[1] + y * brightness)

            weight_totals += brightness

    weight_totals = max(weight_totals, 1)

    # The center of mass is the contribution totals divided by the weight totals
    return (contribution_totals[0] / weight_totals, contribution_totals[1] / weight_totals)

# previews the image which will be sent to the network
def preview_image(img: Image.Image):
    output = parse_image(img)

    output.save("preview.png")

# converts the image to make it similar to all the other data, then turns it into a tensor which can be analyzed by the neural network
def parse_image(img: Image.Image):
    # Initialize a new 28 * 28 image to send into the network
    output = Image.new("RGB", (28, 28), color="black")
    # Resize the drawing down to 20 * 20 to fit the MNIST convention
    img.thumbnail((20, 20), Image.Resampling.BOX)
    center_of_mass = get_center_of_mass(img)
    
    # Calculate the offset so that the center of mass of the image is the center of the output
    center_coord = 13.5 #(28 / 2)
    
    offset_x = round(center_coord - center_of_mass[0])
    offset_y = round(center_coord - center_of_mass[1])

    offset_x = max(min(28 - center_coord, offset_x), 0)
    offset_y = max(min(28 - center_coord, offset_y), 0)

    output.paste(img, (offset_x, offset_y))

    output = output.convert("L")

    return output

def classify(img: Image.Image):
    # convert the image to make it mostly compatible with the model
    inputs = parse_image(img)
    inputs = transforms.ToTensor()(inputs)
    inputs = torch.flatten(inputs)

    # model expects image to be wrapped (i.e. have shape at least (1, 28 * 28))
    inputs = inputs.unsqueeze(0)

    # pass the image to the model
    with torch.no_grad():
        logits = model(inputs)

    tensor_probs = torch.softmax(logits, dim = 1)[0]

    probabilities = {}
    probabilities_string = ""

    for i in range(10):
        probabilities[i] = tensor_probs[i].item()

    for i in range(10):
        key = max(probabilities, key=probabilities.get)
        probabilities_string += f"{key}: {probabilities[key] * 100:.2f}%\n"
        del probabilities[key]

    return probabilities_string