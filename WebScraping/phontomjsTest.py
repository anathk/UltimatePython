from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
import time


#driver = webdriver.PhantomJS(executable_path="C:\\software\\phantomjs\\bin\\phantomjs.exe")

driver = webdriver.Firefox()
driver.get("http://heroes-wow.com/wotlk/index.php?page=login")

driver.implicitly_wait(1)
username_element = driver.find_element_by_name('username')
password_element = driver.find_element_by_name('password')
submit_element = driver.find_element_by_xpath('//input[@type="submit" and @value="log in"]')
time.sleep(2)
username_element.send_keys('test')
password_element.send_keys('test')
time.sleep(6)
submit_element.click()

driver.get('http://topg.org/server-heroes-wow-id347987')
vote_element = driver.find_element_by_partial_link_text('Visit & Play')
vote_element.click()
time.sleep(2)



driver.close()


