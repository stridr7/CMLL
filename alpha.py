import keyboar, time
from selenium                               import webdriver
from webdriver_manager.chrome               import ChromeDriverManager
from selenium.webdriver.chrome.service      import Service
from selenium.webdriver.chrome.options      import Options

#usado unicamente para fazer login no whatsapp
chrome_options = Options()
chrome_options.add_argument(r"user-data-dir=" + "c://users//public//user_data")
driver = webdriver.Chrome(executable_path=r"chromedriver.exe")
driver = webdriver.Chrome(chrome_options=chrome_options)
driver.get("https://web.whatsapp.com/")
time.sleep(40)
key.press_and_release('enter')
key.press_and_release('alt,space')
key.press_and_release('n')
import cmll


