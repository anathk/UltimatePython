from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
import time
import os
import pyautogui


url = 'https://www.icmt.ohio.edu/eio'
driver = webdriver.Chrome()
# driver = driver = webdriver.PhantomJS(executable_path="C:\\software\\phantomjs\\bin\\phantomjs.exe")
driver.get(url)


try:
    print('Start')
    username_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'LoginUser_UserName')))
    password_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'LoginUser_Password')))
    submit_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'LoginUser_LoginButton')))
    username_element.send_keys('test')
    password_element.send_keys('test')
    submit_element.click()
    # print(driver.page_source)

    status_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'ctl00_LinkButtonMyStatus')))
    status_element.click()

    out_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'ctl00_UpdateStatusControluc_DataList1_ctl01_LabelForRadio')))
    in_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'ctl00_UpdateStatusControluc_DataList1_ctl00_LabelForRadio')))
    unknown_return_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'ctl00_UpdateStatusControluc_CheckBoxUnknownReturningTime')))
    confirm_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'ctl00$UpdateStatusControluc$ButtonSave')))

    out_element.click()
    unknown_return_element.click()
    confirm_element.click()

except Exception as e:
    print(e)
finally:
    print('Done')
    # driver.close()
    # driver.quit()


