# -*- coding: utf-8 -*-
'''
Script to download bugs detail from BugZilla of Eclipse.

'''

from io import StringIO

import requests
from lxml import etree

# First URL have a 1000 limit.
#BUG_LIST_URL_PATTERN = 'https://bugs.eclipse.org/bugs/buglist.cgi?component={component}&product={product}&resolution=---'
#BUG_LIST_URL_PATTERN_UNLIMIT = 'https://bugs.eclipse.org/bugs/buglist.cgi?component={component}&limit=0&order=bug_status%2Cpriority%2Cassigned_to%2Cbug_id&product={product}&query_format=advanced&resolution=---'
#BUG_DETAIL_URL_PATTERN = 'https://bugs.eclipse.org/bugs/show_bug.cgi?ctype=xml&id={bug_id}'
#PRODUCT_COMPONENTS_URL_PATTERN = 'https://bugs.eclipse.org/bugs/describecomponents.cgi?product={product_name}'

# Since all the argument of the URLs have to be encoded by the requests,
# so we extract it out from the original URLs.
BUG_LIST_URL_PATTERN = 'https://bugs.eclipse.org/bugs/buglist.cgi'
BUG_DETAIL_URL_PATTERN = 'https://bugs.eclipse.org/bugs/show_bug.cgi'
PRODUCT_LIST_URL_PATTERN = 'https://bugs.eclipse.org/bugs/describecomponents.cgi'
PRODUCT_COMPONENTS_URL_PATTERN = 'https://bugs.eclipse.org/bugs/describecomponents.cgi'


def get_product_names(timeout=10):
    '''This is going to get product names.
    '''

    product_names = []

    res = requests.get(PRODUCT_LIST_URL_PATTERN, timeout=timeout)
    res.encoding = 'utf-8'

    parser = etree.HTMLParser()
    tree = etree.parse(StringIO(res.content.decode()), parser)
    root = tree.getroot()

    for element in root.cssselect("a[href^='describecomponents.cgi?product=']"):
        product_names.append(element.text.replace(u"\u00a0", " ").replace(u"\u2011", "-"))

    return product_names

def get_product_components(product_name, timeout=10):
    '''This get product names in a component.
    '''

    components = []

    payload = {
        'product': product_name
    }
    res = requests.get(PRODUCT_COMPONENTS_URL_PATTERN, params=payload, timeout=timeout)
    res.encoding = 'utf-8'

    parser = etree.HTMLParser()
    tree = etree.parse(StringIO(res.content.decode()), parser)
    root = tree.getroot()

    for element in root.cssselect("a[href^='buglist.cgi']"):
        print(element.text, end=', ')
        components.append(element.text.replace(u"\u00a0", " ").replace(u"\u2011", "-"))

    return components

def get_bug_ids_of_component(product, component, timeout=10):
    '''Get bug ids using product and component name.
    '''

    bug_ids = []

    # Sending request for the bug list.
    # Unlimit bug list payload.
    #?component={component}&limit=0&order=bug_status%2Cpriority%2Cassigned_to%2Cbug_id&product={product}&query_format=advanced&resolution=---
    payload = {
        'component': component,
        'limit': 0,
        #'order': 'bug_status,priority,assigned_to,bug_id', # Leave this out seems fine.
        'product': product,
        #'query_format': 'advanced', # Leave this out seems fine.
        #'resolution': '---' # This limit bugs to be only the resolution=---
    }
    res = requests.get(BUG_LIST_URL_PATTERN, params=payload, timeout=timeout)
    res.encoding = 'utf-8'

    parser = etree.HTMLParser()
    tree = etree.parse(StringIO(res.content.decode()), parser)
    root = tree.getroot()

    # Extract the ids.
    for element in root.cssselect("td[class^='first-child'] a[href^='show_bug']"):
        bug_ids.append(element.text)

    return bug_ids

def get_bug_detail(bug_id, timeout=10):
    '''This is going to get bug detail.
    It returns XML content of the bug.
    '''

    #?ctype=xml&id={bug_id}
    payload = {
        'ctype': 'xml',
        'id': bug_id
    }
    res = requests.get(BUG_DETAIL_URL_PATTERN, params=payload, timeout=timeout)
    res.encoding = 'utf-8'

    return res.content.decode()