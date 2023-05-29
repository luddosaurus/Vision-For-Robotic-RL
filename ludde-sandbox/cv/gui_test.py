import tkinter
import customtkinter as tk
from PIL import ImageTk, Image
import cv2


def display_image(app, image):
    # Rearrange colors
    blue, green, red = cv2.split(image)
    img = cv2.merge((red, green, blue))
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)

    # Create a Label to display the image
    label = tk.CTkLabel(app, image=imgtk, text="")
    label.pack()


tk.set_appearance_mode("System")
tk.set_default_color_theme("green")


app = tk.CTk()
app.geometry("1280x720")
app.title("Calibrator")


image = cv2.imread('test.png')
display_image(app, image=image)

control_info = ['[q]uit', '[s]ave', '[u]ndo', '[d]elete', '[p]lot', '[h]save']


info_text = tk.CTkLabel(app, text="best gui in the world?")
info_text.pack(padx=10, pady=10)

button_row = tk.CTkFrame(app)
button_row.pack()

for i in range(len(control_info)):
    button = tk.CTkButton(button_row, text=control_info[i])
    button.pack(side=tk.LEFT, padx=5)

button_row.pack(anchor=tk.CENTER, padx=20, pady=20)

# Run app
app.mainloop()



