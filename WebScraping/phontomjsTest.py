from selenium import webdriver


driver = webdriver.PhantomJS(executable_path="C:\\software\\phantomjs\\bin\\phantomjs.exe")
driver.get("http://heroes-wow.com/wotlk/index.php?page=login")

driver.implicitly_wait(1)
element = driver.find_element_by_name('username')


