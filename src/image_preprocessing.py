from matplotlib import pyplot as plt

from PIL.ImageDraw import Draw

def draw_dots(draw, coordinates):
    for x, y in coordinates:
        draw.ellipse(((x-5, y-5), (x+5, y+5)), fill='red', outline='red')
        
def draw_rectangle(img, x0, y0, x1, y1, col_dots='orange', col_box='blue'):
    # draw dots
    plt.scatter([x0, x1], [y0, y1], s=100, c=col_dots)
    
    # draw box
    plt.plot([x0, x0], [y0, y1], c=col_box)
    plt.plot([x1, x1], [y0, y1], c=col_box)
    plt.plot([x0, x1], [y0, y0], c=col_box)
    plt.plot([x0, x1], [y1, y1], c=col_box)
    
    plt.imshow(img)