from robobrowser import RoboBrowser
#from fake_useragent import UserAgent
from bs4 import BeautifulSoup




# Browser
#br = mechanize.Browser()
br = RoboBrowser(history=True, user_agent='Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.2 (KHTML, like Gecko) Chrome/22.0.1216.0 Safari/537.2')


# The site we will navigate into, handling it's session
br.open('http://heroes-wow.com/wotlk/index.php?page=login')

login_form = br.get_form(action="http://heroes-wow.com/wotlk/execute.php?take=login")
login_form['username'].value = 'anathk2'
login_form['password'].value = 'wow123456'
login_form['rememberme'].value = '1'

br.submit_form(login_form)


br.open('http://topg.org/server-heroes-wow-id347987')
links = br.find_all('a', href=True)
br.follow_link(links[22])
result = br.parsed

new_links = br.find_all('a', href=True)
br.follow_link(new_links[1])






