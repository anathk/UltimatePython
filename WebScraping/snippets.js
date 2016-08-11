/**
 * Created by zt359416 on 8/11/2016.
 */
driver = webdriver.Firefox()

driver.get("http://stackoverflow.com/questions/7794087/running-javascript-in-selenium-using-python")
driver.execute_script("document.getElementsByClassName('comment-user')[0].click()")
