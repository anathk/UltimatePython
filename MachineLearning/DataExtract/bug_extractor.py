# -*- coding: utf-8 -*-
'''
Bug Extractor Script

This is the script to extract the the bug detail from the BugZilla server.
It is going to build its init_state first. This init_state contains the
information of the product names and their components.

After that the script is going to use these information to download the
bug_ids for one component at a time (this require product name and component
name).

With each bug coming in, script is going to parse the XML into Python object.
It then insert the object into MongoDB.

This script is capable of resumming from the last session. However, it still
have some problem with the state saving and resumming.

Outline of the process:
1.
Check if complete products and components file is exist or not.
If not: build products and components file.

option: --force-rebuild is going to rebuild products and components file.
file-name: products_and_components.json

2.
If products and components are there,
load only one components for processing.

The bug id that are not process in this session is going to be saved to
the current-work.json. The information of the products and components are
going to be saved as well.

file-name: current-work.json

3. Repeat until current-work.json is empty

State file strucuture:
{
    'products': [
        {
            'name': 'Name'
            'components': ['', '', '']
        },
    ],
    'current_bug_ids': ['']
    'current_product': {
        'name': '',
        'components': ['', '', '']
    },
    'current_component': ''
}
'''

import sys
import json
import signal
import argparse
import traceback
from time import sleep
from os.path import join, exists

import requests
from lxml import etree
from pymongo import MongoClient, errors

from lib import parser
from lib import resource

# Variable to store state of the program.
state = {
    'products': [],
    'current_bug_ids': [],
    'current_product': None,
    'current_component': None
}

# Use by interrupt_handler to tell script to
# exit the main loop.
is_running = True

def interrupt_handler(signal, frame):
    '''Handle the signal sent to the script.

    This is going to stop the execution loop, so
    the script can exit cleanly.'''

    global is_running

    print('\n\nInterrupted...\n')
    print('Finishing up the work...', end='')

    is_running = False

def load_state(filename):
    '''This load the data of previous session.

    Since the state file can be in any name, so it
    is required to have filename.'''

    state = None

    with open(filename, 'r') as f:
        state = json.load(f)

    return state

def save_state(state, filename):
    '''Save state of the script.'''

    print('Saving state to {}...'.format(filename), end='')
    with open(filename, 'w') as f:
        json.dump(state, f)
    print('done')

def build_init_state():
    '''This is going to build the init_state.json

    It is going to get add the product and component names.
    After that build a JSON file that contains these information.'''

    product_names = resource.get_product_names(timeout=args['timeout'])

    print('')
    for product_name in product_names:
        print('Getting information for {} : '.format(product_name), end='')

        obj = {}
        obj['name'] = product_name
        obj['components'] = resource.get_product_components(product_name, timeout=args['timeout']);

        state['products'].append(obj)
        print('\n')

        sleep(sleep_time)

    save_state(state, 'init_state.json')

def parse_argument():
    args = {}
    arg_parser = argparse.ArgumentParser(description='Extract bugs\' detail from BugZilla server.')
    arg_parser.add_argument('--state-file', action='store', nargs=1, default='current_state.json',
                            help='Specify the state file.')
    arg_parser.add_argument('-r','--rebuild-initial-state', action='store_true',
                            help='Force the script to rebuild the init_state.json even it exists.')

    arg_parser.add_argument('-u', '--db-url', action='store', nargs=1, default='mongodb://127.0.0.1:27017',
                            help='The database URL for mongoDB.')
    arg_parser.add_argument('-d', '--db-name', action='store', nargs=1, default='dl4se',
                            help='The database name for mongoDB.')
    arg_parser.add_argument('-c', '--db-collection-name', action='store', nargs=1, default='bugs',
                            help='The database collection name for mongoDB.')
    arg_parser.add_argument('--db-update', action='store_true',
                            help='The database collection name for mongoDB.')

    arg_parser.add_argument('-t', '--timeout', action='store', type=float, default=180.0,
                            help='The timeout for the request to the BugZilla server.')
    arg_parser.add_argument('-s', '--sleep-time', action='store', type=float, default=2.0,
                            help='The sleep time between each request to the BugZilla server.')

    return vars(arg_parser.parse_args())

if __name__ == '__main__':
    args = parse_argument()

    db_url = args['db_url']
    db_name = args['db_name']
    db_collection_name = args['db_collection_name']
    timeout = args['timeout']
    sleep_time = args['sleep_time']

    if args['rebuild_initial_state']:
        print('Rebuilding initial state...')
        build_init_state()
        print('')

    if exists(args['state_file']):
        print('Found {state_file}; load previous sesssion data...'.format(state_file=args['state_file']))
        state.update(load_state(args['state_file']))
    else:
        # Use init_state.json as the current state.
        print('No {state_file} found; using init_state.json...'.format(state_file=args['state_file']))

        if not exists('init_state.json'):
            print('No init_state.json found; rebuilding...')
            build_init_state()
            print('')

        state.update(load_state('init_state.json'))

    # Register the signal handler.
    signal.signal(signal.SIGINT, interrupt_handler)

    # Create MongoDB client.
    db_client = MongoClient(db_url)
    db = db_client[db_name]

    try:
        while is_running:

            if state['current_product'] == None and len(state['products']) != 0:
                state['current_product'] = state['products'].pop(0)

            if state['current_component'] == None and (state['current_component'] == None and len(state['current_product']['components']) != 0):
                state['current_component'] = state['current_product']['components'].pop(0)

                print('\n\nLoad the bug ids of [{}][{}]...'.format(state['current_product']['name'], state['current_component']), end='')
                bug_ids = resource.get_bug_ids_of_component(\
                    state['current_product']['name'], \
                    state['current_component'], \
                    timeout=args['timeout'])
                print('done')
                print('Found {} bug_ids.'.format(len(bug_ids)))
                state['current_bug_ids'].extend(bug_ids)

            if state['current_product'] == None and state['current_component'] == None:
                print('Nothing to do')
                break

            # All bug_id in the list is going to be only for one component.
            while len(state['current_bug_ids']) != 0:
                if not is_running:
                    break

                bug_id = state['current_bug_ids'].pop(0)

                print('\r[{product_name}][{component_name}] {number_of_bug_left} bugs left; Currently retriving #{bug_id} '.format(\
                            product_name=state['current_product']['name'], \
                            component_name=state['current_component'], \
                            number_of_bug_left=len(state['current_bug_ids']),
                            bug_id=bug_id), end='')

                content = resource.get_bug_detail(bug_id, timeout=args['timeout'])

                # Write raw content to file.
                # with open(join('xml_files', bug_id + '.xml'), 'w') as f:
                    # f.write(content)

                # Parse it from XML to Python dict.
                parsed_object = parser.parse(content)

                # with open(join('xml_files', bug_id + '.json'), 'w') as f:
                #     json.dump(result, f, sort_keys=True, indent=4)

                # Save data to the DB.
                parsed_object['_id'] = parsed_object['bug_id']

                # Remove all attachment data.
                for attachment in parsed_object.get('attachments', []):
                    attachment.pop('data', None)

                try:
                    if args['db_update']:
                        result = db[db_collection_name].update({'_id': parsed_object['bug_id']}, parsed_object, upsert=True)
                    else:
                        result = db[db_collection_name].insert_one(parsed_object)
                except errors.DuplicateKeyError:
                    print('warning: The key is duplicate.')
                    pass

                sleep(sleep_time)

            if len(state['current_bug_ids']) == 0:
                state['current_component'] = None

            if len(state['current_product']['components']) == 0:
                state['current_product'] = None

        # For the 'Finishing up the work...'
        print('done')

    except requests.exceptions.Timeout:
        print('\nTimeout in one of the requests.')

        # Add the information back.
        state['current_bug_ids'].insert(0, bug_id)

    except Exception:
        print('\nUnexpected error encountered.')
        traceback.print_exc()

        # Add the information back.
        state['current_bug_ids'].insert(0, bug_id)

    # Save the session if we still have some work.
    save_state(state, args['state_file'])