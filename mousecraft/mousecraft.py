import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess
import sys
import os

# Path to the main GUI script
MAIN_GUI_SCRIPT = 'mousecraft/gui.py' # why this works ? check this if at some point it doesn't

class StartupWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Mousecraft')
        self.geometry('1100x800')
        self.resizable(False, False)
        self.configure(bg='white')

        # Load background image (mouse.png)
        bg_path = 'mousecraft/resources/mouse.png'
        if os.path.exists(bg_path):
            img = Image.open(bg_path)
            img = img.resize((1100, 800), Image.Resampling.LANCZOS)
            self.bg_img = ImageTk.PhotoImage(img)
            self.bg_label = tk.Label(self, image=self.bg_img)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        else:
            self.bg_label = None

        # Place MouseCraft.png as logo at the top, centered
        logo_path = 'mousecraft/resources/MouseCraft.png'
        if os.path.exists(logo_path):
            logo_img = Image.open(logo_path)
            # Resize logo to fit nicely at the top (e.g., width 400)
            logo_width = 400
            aspect = logo_img.height / logo_img.width
            logo_img = logo_img.resize((logo_width, int(logo_width * aspect)), Image.Resampling.LANCZOS)
            self.logo_imgtk = ImageTk.PhotoImage(logo_img)
            self.logo_label = tk.Label(self, image=self.logo_imgtk, bg='#ffffff', borderwidth=0, highlightthickness=0)
            self.logo_label.place(relx=0.5, y=30, anchor='n')
        else:
            self.logo_label = None

        # Play button
        self.play_btn = tk.Button(self, text='Play', font=('Arial', 24, 'bold'), width=12, command=self.launch_main_gui, bg='#4CAF50', fg='white', activebackground='#388E3C', activeforeground='white')
        self.play_btn.place(relx=0.5, rely=0.65, anchor='center')

        # Exit button
        self.exit_btn = tk.Button(self, text='Exit', font=('Arial', 24, 'bold'), width=12, command=self.exit_popup, bg='#F44336', fg='white', activebackground='#B71C1C', activeforeground='white')
        self.exit_btn.place(relx=0.5, rely=0.75, anchor='center')

    def launch_main_gui(self): # launch main when this one closes 
        """Close startup window; main GUI will be launched by gui.py"""
        self.destroy()

    def exit_popup(self):
        # Show popup that prevents exit
        popup = tk.Toplevel(self)
        popup.title('No Exit')
        popup.geometry('350x120')
        popup.resizable(False, False)
        popup.grab_set()
        label = tk.Label(popup, text='Please choose Play instead', font=('Arial', 16))
        label.pack(pady=20)
        close_btn = tk.Button(popup, text='Close', font=('Arial', 14), command=popup.destroy)
        close_btn.pack(pady=5)
        popup.protocol('WM_DELETE_WINDOW', lambda: None)  # Disable X button


if __name__ == '__main__':
    app = StartupWindow()
    app.mainloop()
