from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
import time


#driver = webdriver.PhantomJS(executable_path="C:\\software\\phantomjs\\bin\\phantomjs.exe")

driver = webdriver.Firefox()
driver.get("http://heroes-wow.com/wotlk/index.php?page=login")

try:
    username_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, 'username')))
    password_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, 'password')))
    submit_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//input[@type="submit" and @value="log in"]')))
    username_element.send_keys('test')
    password_element.send_keys('test')
    submit_element.click()

finally:
    pass


driver.get('http://topg.org/server-heroes-wow-id347987')
vote_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, 'Visit & Play')))
#vote_element = driver.find_element_by_partial_link_text('Visit & Play')
vote_element.click()
handles = driver.window_handles
driver.switch_to.window(handles[-1])
#print(driver.page_source)
driver.implicitly_wait(2)
last_element = driver.find_element_by_css_selector('a.wotlk')
print(last_element)
#last_element = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CLASS_NAME, 'button wotlk')))
last_element.click()
#driver.close()


