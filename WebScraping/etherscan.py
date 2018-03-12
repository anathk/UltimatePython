from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os
import pathlib
import requests
from lxml import html
import logging
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler('etherscan.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(ch)
logger.addHandler(fh)

# from pyvirtualdisplay import Display
import time
import clipboard



def save_code_to_file(filename, content, path):
    try:
        print(type(content))
        path = path / filename
        path.touch()
        path.write_text(content, encoding='utf-8')
        # Add suffix if needed.
        # path.rename(path.with_suffix('.txt'))
    except Exception as e:
        logger.error(f'Error on {filename}.')
        logger.error(e)
        #logger.error(content)

def extract_links(contracts_url):
    '''
    :param contracts_url: a page contains multiple verified contracts
    :return: next page of contracts_url and contract_links(list) of current page.
    '''
    # ua = UserAgent().random
    # r = requests.get(contracts_url, headers={'User-Agent': ua})
    # print(r.status_code)
    driver = webdriver.Chrome(executable_path="C:\\software\\chromedriver_win32\\chromedriver.exe",
                              chrome_options=options)
    driver.get(contracts_url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '/html/body/div[1]/div[4]/div[2]/div[2]/p/a[1]')))
    doc = html.fromstring(driver.page_source)
    next_page = doc.xpath('/html/body/div[1]/div[4]/div[4]/div/p/a[3]/@href')[0]
    contract_links = doc.xpath('/html/body/div[1]/div[4]/div[3]/div/div/div/table/tbody/tr/td[1]/a/@href')
    driver.quit()
    return contract_links


def webdriver_copy(contract_url):
    driver = webdriver.Chrome(executable_path="C:\\software\\chromedriver_win32\\chromedriver.exe",
                              chrome_options=options)
    driver.get(contract_url)
    try:
        source_button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="dividcode"]/span[1]/button')))
        source_button = driver.find_element_by_xpath('//*[@id="dividcode"]/span[1]/button')
        source_button.click()
        code = clipboard.paste()
    except Exception as e:
        logger.exception(f'Exception on {contract_url}')
        logger.exception(e)
        code = None

    #driver.close()
    driver.quit()
    return code

# display = Display(visible=0, size=(800, 600))
# display.start()

# -------------------------------Chrome Options--------------------------------------------
# options = webdriver.ChromeOptions()
options = Options()
# options.add_argument('--headless')
# options.add_argument("--window-size=1920x1080")
options.add_argument('--disable-gpu')
# options.add_argument("--test-type")
# options.add_argument("--start-maximized")
options.add_argument("--start-minimized")
options.add_argument("--ignore-certificate-errors")
# options.add_argument("--disable-popup-blocking")
options.add_argument("--incognito")
options.add_argument("--disable-infobars")
# -------------------------------End Chrome Options----------------------------------------

# driver = webdriver.PhantomJS(executable_path="C:\\software\\phantomjs\\bin\\phantomjs.exe")

source_code_path = pathlib.Path('C:/Users/zt424616/Documents/github/UltimatePython/WebScraping/contract_code')
domain_url = 'https://etherscan.io'
contracts_url = 'https://etherscan.io/contractsVerified/'
contract_url = 'https://etherscan.io/address/0x7f288ff5a8055f5f6103a80dd806cf8415e035c7#code'
contract_op_base_url = 'https://etherscan.io/api?module=opcode&action=getopcode&address='

#driver = webdriver.Chrome()


'''
r = requests.get('https://etherscan.io/address/0xd2d121383187d4ed9cb4d40e7ce5d676f5994e0d#code')
doc = html.fromstring(r.content)

#'https://etherscan.io/api?module=opcode&action=getopcode&address=0xd2D121383187D4Ed9Cb4D40E7cE5D676f5994e0D'

contract_creation_code = doc.xpath('//*[@id="verifiedbytecode2"]/text()')[0]
print(contract_creation_code)

contract_op_code = requests.get('https://etherscan.io/api?module=opcode&action=getopcode&address=0xd2D121383187D4Ed9Cb4D40E7cE5D676f5994e0D')
print(contract_op_code.json()['result'].replace('<br>', '\n'))
'''

for page_num in range(464, 675):
    page_num += 1
    current_page = contracts_url + str(page_num)
    logger.debug(f'Crawling on page {current_page}')
    contract_links = extract_links(current_page)

    for contract_link in contract_links:
        logger.debug(f'Crawling on contract: {contract_link}')
        contract_address = contract_link.replace('#', '/').split('/')[2]
        contract_url = domain_url + contract_link
        contract_op_url = contract_op_base_url + contract_address

        source_code = webdriver_copy(contract_url)
        if source_code is not None:
            save_code_to_file(contract_address, source_code, source_code_path)
        else:
            continue

#print(doc)

# Test for capture, works under headless mode.
# driver.get("https://www.google.com")
# lucky_button = driver.find_element_by_css_selector("[name=btnI]")
# lucky_button.click()
# driver.get_screenshot_as_file("capture.png")


# driver.get(contract_url)
# source_button = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="dividcode"]/span[1]/button')))
# source_button = driver.find_element_by_xpath('//*[@id="dividcode"]/span[1]/button')
# source_button.click()
# text = clipboard.paste()

#time.sleep(5)
# driver.close()
# display.stop()
# driver.quit()
#save_code_to_file('0x30b0d22416075184b8111c0e033fe6f33e6b7996', text, source_code_path)
print('Done!')


