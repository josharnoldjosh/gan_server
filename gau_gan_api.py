from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import os
import base64
import time
from io import BytesIO
from PIL import Image
from pyvirtualdisplay import Display

class GauGan:
    def __init__(self):
        self.display = Display(visible=0, size=(4000, 2500))  
        self.display.start()
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        self.driver = webdriver.Chrome('/usr/local/bin/chromedriver', options=chrome_options)                        
        self.driver.get("http://34.216.122.111/gaugan/")
        self.driver.find_element_by_id("myCheck").click()
        self.previous_output = "" 

    def render(self, image_name, style_name="example3"):        
        self.driver.find_element_by_id('segmapfile').send_keys(os.getcwd()+'/'+image_name)
        self.driver.find_element_by_id('btnSegmapLoad').click()
        self.driver.find_element_by_id(style_name).click()
        self.driver.find_element_by_id("render").click()        

    def should_save_again(self, output_name):
        if self.previous_output != "" and os.stat(output_name).st_size == os.stat(self.previous_output).st_size:
            return True
        elif os.stat(output_name).st_size < 6000:
            return True
        else:
            self.previous_output = output_name

    def save(self, output_name="output.jpg"):
        canvas = self.driver.find_element_by_id('output')
        canvas_base64 = self.driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);", canvas)
        self.canvas_png = base64.b64decode(canvas_base64)
        with open(output_name, 'wb') as f:
            f.write(self.canvas_png)

    def image_data(self):                    
        image = Image.open(BytesIO(self.canvas_png))
        buffered = BytesIO()
        image.save(buffered, format="png")                
        return 'data:image/png;base64,'+base64.b64encode(buffered.getvalue()).decode('ascii')    

    def close(self):
        self.driver.close()

# GG = GauGan()

# GG.render('AABM6PX5IIA0AAABM6PX5IIA0A3QY7M81QH7MO5OD78ZNIF60XH3OK7C37WLF8U1WPQ51OZ2PPIQ9S01MS4K6D_0_raw_label_smoothed.png', "snow")
# GG.save_image("snow.jpg")

# GG.render('AABM6PX5IIA0AAABM6PX5IIA0A3QY7M81QH7MO5OD78ZNIF60XH3OK7C37WLF8U1WPQ51OZ2PPIQ9S01MS4K6D_0_raw_label_smoothed.png', "sunset")
# str_result = GG.save_image("sunset.jpg")

# GG.close()