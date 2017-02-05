# -*- coding: utf-8 -*-
'''
The parser module.

Planning:

stack
curr_obj

stack_item = {
    'tag_name': '',
    'data': object
}

if tag starts
    if the current_obj is not None
        push the current_obj into stack

    new current_obj

    if len(attrib) != 0:
        add it to the current_obj

if data
    if curr_obj is None || if the stripped text == '\n'
        ignore the data.

if tag ends
    if the top of stack is the list type
        append the current_obj to it.
    else:
        pop it out from stack
        append it to the main obj

bug = {
    'id': '',
    'other-attrib': '',
    'attachments': [
        {
            '':
        }
    ],
    'comments': [
        {
            '':
        }
    ],
}

if new tag
    create new obj
    add it to the larger obj
    the ref to new obj should point to the one inside

if tag end
'''

from os import path
import json

from lxml import etree

class BugZillaTarget(object):
    '''Class that contains how to parse a XML file from BugZilla server.'''

    def __init__(self):
        '''Create some variables, and initialize the stack.'''

        self.stack = []
        self.top_level_object  = {
            'comments': [],
            'attachments': [],
            'cc_list': []
        }
        self.current_object = None


    def start(self, tag, attrib):
        '''This is called when the start of a tag is encountered.

        It will determine weather or not it should allocate some space in
        top level object.'''

        if self.current_object != None:
            self.stack.append(self.current_object)

        # New stack item.
        # - tag_name
        # - tag_object
        self.current_object = {
            'tag_name': tag,
            'tag_object': None
        }

        if len(self.stack) >= 1 and tag in ['long_desc']:
            self.top_level_object['comments'].append({})

        if len(self.stack) >= 1 and tag in ['attachment']:
            self.top_level_object['attachments'].append({})

        # Initialize the tag_object.
        if len(attrib) != 0:
            # Init tag_object as dict
            # and store the attributes.
            self.current_object['tag_object'] = {
                'attributes': []
            }

            for attrib in attrib.items():
                self.current_object['tag_object']['attributes'].append(attrib)

        else:
            # Since we do not have the attributes,
            # we are going to use str type as a placeholder.
            self.current_object['tag_object'] = ''


    def end(self, tag):
        '''This is called when tag is end.

        It will determine where to put the current_object into
        the top_level_object. It use stack information as part of the dicision
        making process.'''

        #print('end ' + tag)
        #print('top: ' + self.stack[-1]['tag_name'])

        if tag in ['long_desc', 'bug', 'attachment', 'bugzilla']:
            self.current_object = None
            return

        # Check the top of the stack for a container type tag.
        # And it is not the closure of itself.
        if self.stack[-1]['tag_name'] != tag and \
            self.stack[-1]['tag_name'] in ['long_desc', 'bug', 'attachment']:

            if self.stack[-1]['tag_name'] == 'long_desc':
                #print(self.current_object)
                self.top_level_object['comments'][-1][self.current_object['tag_name']] = self.current_object['tag_object']
            elif self.stack[-1]['tag_name'] == 'attachment':
                self.top_level_object['attachments'][-1][self.current_object['tag_name']] = self.current_object['tag_object']
            elif self.current_object['tag_name'] == 'cc':
                self.top_level_object['cc_list'].append(self.current_object['tag_object'])
            else:
                self.top_level_object[self.current_object['tag_name']] = self.current_object['tag_object']

            self.current_object = None
        else:
            object_from_stack = self.stack.pop()
            self.top_level_object[object_from_stack['tag_name']] = object_from_stack['tag_object']


    def data(self, data):
        '''This is called when the data in tag is encountered.

        The data is put into the current_object.'''

        if self.current_object == None:
            return

        if type(self.current_object['tag_object']) is dict:
            if not ('data' in self.current_object['tag_object'].keys()):
                self.current_object['tag_object']['data'] = ''

            self.current_object['tag_object']['data'] += data
        elif type(self.current_object['tag_object']) is str:
            self.current_object['tag_object'] += data

    def comment(self, text):
        '''The comment is currenly ignore, but printed out.'''

        print('comment: ' + text)

    def close(self):
        '''This is called when the parsing is complete.

        It then return the parsed content back.'''

        return self.top_level_object

# The parser object.
parser = etree.XMLParser(target = BugZillaTarget(), huge_tree=True)

def parse(xml_content):
    '''Parse the content in XML to Python object.

    string of XML text is required.'''

    # The encoding information have to be removed.
    xml_content = '\n'.join(xml_content.split('\n')[1:])
    # The following line prevent malform character to be fed into the parser.
    xml_content = xml_content.replace('\0', '')

    return etree.XML(xml_content, parser)