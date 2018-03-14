import requests
import shutil # high-level operations on files and collections of files
from lxml import html
from datetime import datetime
import hashlib
from fake_useragent import UserAgent
from pathlib import Path
import hashlib
from datetime import datetime
import logging
import os
import multiprocessing

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler('apkpure.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(ch)
logger.addHandler(fh)


save_path = Path('E:/apk_pure_games')
ua = UserAgent()
headers = {'User-Agent': ua.chrome}
url = 'https://apkpure.com/facebook-messenger/com.facebook.orca/download?from=category'

base_url = 'https://apkpure.com'
app_url = 'https://apkpure.com/app'

# Apps
# //*[@id="pagedata"]/li[1]/div[4]/a
# //*[@id="pagedata"]/li[2]/div[4]/a
# //*[@id="pagedata"]/li[20]/div[4]/a

session = requests.Session()
session.headers.update(headers)

r = session.get(app_url)
doc = html.fromstring(r.content)
categories_apps = doc.xpath('/html/body/div[2]/div[2]/div/div[2]/div[2]/ul/li/a/@href')[::-1]
categories_games = doc.xpath('/html/body/div[2]/div[2]/div/div[2]/div[1]/ul/li/a/@href')
logger.debug(f'categories_apps: {categories_apps}')
logger.debug(f'categories_games: {categories_games}')

#categories_apps = ['/house_and_home', '/health_and_fitness', '/food_and_drink', '/finance', '/events', '/entertainment', '/education', '/dating', '/communication', '/comics', '/business', '/books_and_reference', '/beauty', '/auto_and_vehicles', '/art_and_design']
categories_games = categories_games[1:]
def process_page(category_url):
    logger.debug(f'Crawling on page: {category_url}')
    r = session.get(category_url)
    doc = html.fromstring(r.content)
    app_urls = doc.xpath('//*[@id="pagedata"]/li/div[4]/a/@href')
    if app_urls is not None and len(app_urls) > 0:
        process_app_urls(app_urls)

def test_download(session, download_link):
    with session.get(download_link, stream=True) as r:
        size = sum(len(chunk) for chunk in r.iter_content(8196))
        logger.debug(f'{str(datetime.now())} File size: {size} bytes')

def process_app_urls(app_urls):
    for app_url in app_urls:
        app_url = base_url + app_url
        logger.debug(f'{str(datetime.now())} Crawling on app: {app_url}')
        r = session.get(app_url)
        doc = html.fromstring(r.content)
        logger.debug(f'Status code: {r.status_code}')

        try:
            download_link = doc.xpath('//*[@id="download_link"]/@href')[0]
            logger.debug(f'{str(datetime.now())} Download link: {download_link}')
            # p = multiprocessing.Process(target=test_download, args=(session, download_link))
            # p.start()
            # p.join(100)
            #
            # if p.is_alive():
            #     logger.debug(f'{str(datetime.now())} Large file discarded.')
            #     p.terminate()
            #     p.join()
            # else:
            size = doc.xpath('/html/body/div[2]/div[1]/div[2]/div[2]/h1/span[2]/span/text()')[0]
            if float(size.split()[0].replace('(', '')) > 20 or size.split()[1].replace(')', '') == 'GB':
                logger.debug(f'{str(datetime.now())} File size: {size}, discarded.')
            else:
                logger.debug(f'{str(datetime.now())} File size: {size}, Start downloading...')
                with session.get(download_link, stream=True) as r:
                    # size = sum(len(chunk) for chunk in r.iter_content(8196))
                    # logger.debug(f'{str(datetime.now())} File size: {size} bytes')

                    # print(type(r.raw))
                    filename = save_path / hashlib.sha1(download_link.encode('utf-8')).hexdigest()
                    with open(filename, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                    logger.debug(f'{str(datetime.now())} {filename} saved successfully. size: {os.stat(filename).st_size} bytes')

        except Exception as e:
            logger.error(f'Error on crawling app: {app_url}')
            logger.error(e)

for category in categories_games:
    for page_num in range(1, 50):
        category_url = base_url + category + '?page=' + str(page_num)
        process_page(category_url)


session.close()
print('Done')