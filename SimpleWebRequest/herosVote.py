import requests

def print_response(r):
    if r.history:
        print ("Request was redirected")
        for resp in r.history:
            print (resp.status_code, resp.url)
        print ("Final destination:")
        print (r.status_code, r.url)
    else:
        print ("Request was not redirected")

#'http://topg.org/server-heroes-wow-id347987'
myData = dict(username='anathk', password='wow123456')
topg_cookie = {'HTTP_REFERER': 'http://topg.org/server-heroes-wow-id347987'}
heros_cookie = {}
url1 = 'http://heroes-wow.com/wotlk/index.php?page=login'
url = 'http://heroes-wow.com/wotlk/execute.php?take=login'
url2 = 'http://www.rpg-paradize.com/site-heros+wow+548+and+335a+225+level-22237'
session = requests.session()

response = session.post(url, data=myData)
print(session.cookies.get('heroes-wow_hash'))
topg_cookie['heroes-wow_hash'] = session.cookies.get('heroes-wow_hash')
response = session.get(url2)
print_response(response)


print(response.cookies)
response = session.get('http://heroes-wow.com/', cookies=topg_cookie)
print_response(response)
response = session.get('http://heroes-wow.com/wotlk/execute.php?take=vote&site=5')
print_response(response)






