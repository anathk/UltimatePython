# This script was originally from Xin Ye
# The original goal was to download issue reports and parse them and then store them in different fields in a MySQL database for future processing.
# This script needs an input file containing the bug-ID.



import urllib
import types
import datetime
# import re
from bs4 import BeautifulSoup

add_entry = ("INSERT INTO UI "
             "(bug_id, summary, description, report_time, reporter, assignee, status, product, "
             "component, importance, commit, author, commit_time, log, files) "
             "VALUES (%(bug_id)s, %(summary)s, %(description)s, %(report_time)s, %(reporter)s, "
             "%(assignee)s, %(status)s, %(product)s, %(component)s, %(importance)s, "
             "%(commit)s, %(author)s, %(commit_time)s, %(log)s, %(files)s)")
time_format = '%Y-%m-%d %H:%M:%S'
# pattern = re.compile(u'[^\u0000-\uD7FF\uE000-\uFFFF]', re.UNICODE)
input_file = 'ui_commit_bug.txt'
url_prefix = 'https://bugs.eclipse.org/bugs/show_bug.cgi?id='
fs = open(input_file, 'r')
line = fs.readline()
while (line):
    ids = line.split()
    commit = ids[0]
    bug = ids[1]
    url = url_prefix + bug
    response = urllib.urlopen(url)
    html = response.read()
    soup = BeautifulSoup(html)
    summary = (soup.title.string).strip()
    descriptions = soup.findAll(name='pre', attrs={'class': 'bz_comment_text'}, limit=1)
    if len(descriptions) == 1:
        description = descriptions[0].string
        if isinstance(description, types.NoneType) == False:
            description = description.strip()
        # description = pattern.sub(u'\uFFFD', description)
        else:
            description = ''
    else:
        description = ''
    report_times = soup.findAll(name='span', attrs={'class': 'bz_comment_time'}, limit=1)
    if len(report_times) == 1:
        report_time = report_times[0].string
        report_time = report_time.strip()
        tmp = report_time.split()
        report_time = tmp[0] + ' ' + tmp[1]
        report_time = datetime.datetime.strptime(report_time, time_format)
        reporter = report_times[0].find_previous(name='span', attrs={'class': 'fn'})
        if isinstance(reporter, types.NoneType) == False:
            reporter = (reporter.string).strip()
        else:
            reporter = ''
    else:
        report_time = datetime.datetime.now()
        reporter = ''
    assignees = soup.findAll(name='span', attrs={'class': 'fn'}, limit=1)
    if len(assignees) == 1:
        assignee = assignees[0].string
        if isinstance(assignee, types.NoneType) == False:
            assignee = assignee.strip()
        else:
            assignee = ''
    else:
        assignee = ''
    statuss = soup.findAll(name='span', attrs={'id': 'static_bug_status'}, limit=1)
    if len(statuss) == 1:
        status = statuss[0].string
        if isinstance(status, types.NoneType) == False:
            status = status.strip()
            tmp = status.split()
            status = tmp[0]
            for i in range(len(tmp) - 1):
                status = status + ' ' + tmp[i + 1]
        else:
            status = ''
    else:
        status = ''
    products = soup.findAll(name='td', attrs={'id': 'field_container_product'}, limit=1)
    if len(products) == 1:
        product = products[0].string
        if isinstance(product, types.NoneType) == False:
            product = product.strip()
        else:
            product = ''
    else:
        product = ''
    components = soup.findAll(name='td', attrs={'id': 'field_container_component'}, limit=1)
    if len(components) == 1:
        component = components[0].string
        if isinstance(component, types.NoneType) == False:
            component = component.strip()
        else:
            component = ''
    else:
        component = ''
    importance = ((soup.find(name='span', attrs={'id': 'votes_container'})).find_previous(text=True))
    if isinstance(importance, types.NoneType) == False:
        importance = importance.strip()
        tmp = importance.split()
        importance = tmp[0]
        for i in range(len(tmp) - 1):
            importance = importance + ' ' + tmp[i + 1]
    else:
        importance = ''
    print(summary)
    data_entry = {
        'bug_id': bug,
        'summary': summary,
        'description': description,
        'report_time': report_time,
        'reporter': reporter,
        'assignee': assignee,
        'status': status,
        'product': product,
        'component': component,
        'importance': importance,
        'commit': commit,
        'author': '',
        'commit_time': report_time,
        'log': '',
        'files': '',
    }
    line = fs.readline()

fs.close()
