import scrapy
from scrapy.spiders import CrawlSpider, Rule, Spider
from scrapy.linkextractors import LinkExtractor
from BugzillaExtract.items import BugzillaextractItem

class BugzillaSpider(CrawlSpider):
    name = 'BugzillaSpider'
    allowed_domains = ['bug.eclipse.org']
    start_urls = ['https://bugs.eclipse.org/bugs/buglist.cgi?component=Ant&product=Platform&resolution=---']

    # allowed_domains = ["sfbay.craigslist.org"]
    # start_urls = ["http://sfbay.craigslist.org/search/npo"]

    # 1 platform link
    # https://bugs.eclipse.org/bugs/describecomponents.cgi?product=Platform
    # 2 components links
    # https://bugs.eclipse.org/bugs/buglist.cgi?component=Compare&product=Platform&resolution=---
    # https://bugs.eclipse.org/bugs/buglist.cgi?component=Ant&product=Platform&resolution=---
    # 3 bug link
    # https://bugs.eclipse.org/bugs/show_bug.cgi?id=92942


    rules = (
        # Extract links matching 'category.php' (but not matching 'subsection.php')
        # and follow links from them (since no callback means follow=True by default).
        #Rule(LinkExtractor(allow=('show_bug')), callback='parse_item'),
        Rule(LinkExtractor(allow=(), restrict_xpaths=('//@href',)), callback="parse_item", follow=True),
    )

    def parse_item(self, response):
        print("hello world")
        print(response)
        self.logger.info('Hi, this is an item page! %s', response.url)
        item = BugzillaextractItem()
        #item['id'] = response.xpath('//td[@id="item_id"]/text()').re(r'ID: (\d+)')
        #item['name'] = response.xpath('//td[@id="item_name"]/text()').extract()
        #item['description'] = response.xpath('//td[@id="item_description"]/text()').extract()
        title = response.xpath('//title/text()').extract()
        print("Title is: "+title)
        # response.xpath('//h1/text()')[0].extract()
        item['title'] = response.xpath('//title/text()').extract()
        return item