#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from lxml import etree as ET
import numpy as np
import argparse
import os
import subprocess
import stat
import shlex
import shutil
import sys
import time
import copy
import datetime
import atexit
import re


# safe guard
if "AUGEROFFLINEROOT" not in os.environ:
    sys.exit("Environment variable AUGEROFFLINEROOT not set! Aborting ...")


# wieso auch immer so, aber muss...
XLINK_NAMESPACE = "http://www.auger.org/schema/types"
XLINK = "{%s}" % XLINK_NAMESPACE

XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"
XSI = "{%s}" % XSI_NAMESPACE


def set_module_options_recursive(node, option_name, value, unit, parents):
    if(len(parents) > 0):
        parent = parents.pop(0)
        c = node.find("./" + str(parent))
        if(c is None):
            c = ET.SubElement(node, str(parent))
        set_module_options_recursive(c, option_name, value, unit, parents)
        return 0
    else:
        c = node.find("./" + str(option_name))
        if(c is None):
            c = ET.SubElement(node, str(option_name))
        c.text = str(value)
        if(unit is not None):
            c.set("unit", str(unit))
        return 0


def setModuleOptions(bootstrap_root, module_name, module_options):
    """ module_options is an array of arrays with the form
    row = [moduleOption, newOptionValue, newOptionUnit, parents]
    newOptionUnit can be None
    parents are a list of parents, if no "parent" of option is the module
    option root node parents should be an empty list.
    """
    bootstrapOverride = bootstrap_root.find("parameterOverrides")
    node = bootstrapOverride.find(".//" + module_name)
    if(node is None):
        a = ET.SubElement(bootstrapOverride, "configLink", {"id": module_name})
        node = ET.SubElement(a, module_name)

    for option_name, value, unit, parents in module_options:
        set_module_options_recursive(node, option_name, value, unit, parents)


def getFormattedBootstrap(bootstrap_string):
    # return string which is written in the bootstrap file

    # removes all links ...
    xmllinks = []
    last_position = 0
    while True:
        pos1 = bootstrap_string.find("&", last_position)
        pos2 = bootstrap_string.find(";", pos1)

        if (pos1 == -1) or (pos2 == -1):
            break

        if (pos2 - pos1) > 50:
            print("warning: pos2 - pos1 > 50, this should not happen")
            print(bootstrap_string[pos1:pos2 + 1])
            break

        if bootstrap_string[pos2 + 1:pos2 + 2] != "/":
            xmllinks.append(bootstrap_string[pos1:pos2 + 1])
            bootstrap_string = bootstrap_string.replace(xmllinks[-1], "")
            last_position = pos1
        else:
            last_position = pos2

    tmp_parser = ET.XMLParser(remove_blank_text=True, remove_comments=False, attribute_defaults=False, recover=False, resolve_entities=True)

    tmp_root_tree = ET.fromstring(bootstrap_string.encode(), tmp_parser)
    bootstrap_string = ET.tostring(tmp_root_tree.getroottree(), pretty_print=True, xml_declaration=True, encoding=docinfo.encoding)

    # adds all links again
    pos1 = bootstrap_string.find(b"<centralConfig>")
    tmp = b"\n"
    for xmllink in xmllinks:
        tmp += xmllink.encode() + b"\n"
    bootstrap_string = bootstrap_string[:pos1 - 1] + tmp + bootstrap_string[pos1 - 1:]
    return bootstrap_string


def parse_cmd_argument_module_option(module_options_str):
    # return empty lists when not set
    print("the following module options will be set in the bootstrap file:")

    module_names = []
    module_options = []

    if args['module_options']:
        for element in module_options_str:
            tmp = element.split("::")
            module_names.append(tmp[0])

            parents = tmp[1:-2]

            tmp_module_value = tmp[-1].split("[")
            module_value = tmp_module_value[0]
            if(len(tmp_module_value) == 1):
                module_value_unit = None
            elif(len(tmp_module_value) == 2):
                if(tmp_module_value[1][-1] == "]"):
                    module_value_unit = tmp_module_value[1][:-1]
                else:
                    print("ERROR: option unit not specified correctly (missing ])")
                    sys.exit(-1)
            else:
                print("ERROR: option unit not specified correctly (too many [)")
                sys.exit(-1)

            module_options.append([tmp[-2], module_value, module_value_unit, parents])
            print("\tmodule: %s, option: %s, value: %s, unit: %s, parents: %s" % (tmp[0], tmp[-2], module_value, module_value_unit, str(parents)))

    return module_names, module_options


''' parse command line arguments '''
parser = argparse.ArgumentParser(description='Generate EventFileReader.xml out of command line options.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataFiles', dest='data_files', metavar='file1', type=str,
                    nargs='+', help='list of input data files, if this option is not specified the EventFileReader.xml specified in the boostrap file is used.')

parser.add_argument('-b', dest='bootstrap', metavar='/path/to/bootstrap.xml',
                    default='bootstrap.xml', type=str, help='bootstrap.xml file')

parser.add_argument('--fileType', metavar='filetype', dest='data_file_type',
                    default='Offline', help='input file type, choose between Offline FDAS CDAS IoAuger CORSIKA CONEX CONEXRandom AIRES SENECA REAS RadioSTAR RadioMAXIMA RadioAERA RadioAERAroot ')

parser.add_argument('--outputPath', dest='output_path', metavar='/output/path/',
                    default='output', type=str, help='output path')

parser.add_argument('--ADSTOutputPath', dest='adst_output_path', metavar='/output/path/',
                    default=False, type=str, help='output path')

parser.add_argument('--outputFileName', dest='output_file_name', metavar='ADST.root',
                    type=str, default='ADST',
                    help='filename of ADST output file, the given name will be appended by "_$uuid.root"')

parser.add_argument('--userAugerOffline', dest='user_auger_offline', nargs='?', metavar='./userAugerOffline',
                    default='AugerOffline', const='./userAugerOffline', type=str, help='path to Offline executable (uses ./userAugerOffline if --userAugerOffline is set without any argument)')

parser.add_argument('--uuid', dest='uuid', metavar='jobid', type=str, default='0001', help='the job uuid')

parser.add_argument('-j', dest='n_threads', metavar='nThreads', type=int, default=1,
                    help='number of cores to use')

parser.add_argument('--moduleOption', dest='module_options', metavar='', nargs='+',
                    type=str, default=None, help='Module options that should be overridden in the bootstrap file. \
                    The syntax is \"ModuleName::ModuleOption::Value[unit]\". The specification of a unit is optional. \
                    \nIf you have a level of recursion in your module options you can specify it with \
                    \"ModuleName::ModuleOptionLevel1::ModuleOptionLevel2::Value[unit]\". \
                    If several jobs are created, an index in the job option can be specified with \"{i}\".')

args = vars(parser.parse_args())
data_file_type = args['data_file_type']
bootstrap_path = args['bootstrap']

# set orig directory of bootstrap
if re.search("/", bootstrap_path):
    bootstrap_dir = os.path.join(*bootstrap_path.split("/")[:-1]) + "/"
else:
    bootstrap_dir = "./"

output_path = os.path.abspath(args['output_path'])

if not os.path.exists(output_path):
    print("ERROR: ouputpath \"%s\" does not exist" % output_path)
    sys.exit(-1)

adst_output_path = args['adst_output_path']
if adst_output_path:
    print("setting adst output path to %s" % adst_output_path)
    adst_output_path = os.path.abspath(adst_output_path)
else:
    adst_output_path = output_path

user_auger_offline = args['user_auger_offline']
_uuid = args['uuid']

module_names, module_options = parse_cmd_argument_module_option(args['module_options'])

n_threads = args['n_threads']
if n_threads > 1:
    if(args["data_files"] is None):
        print("ERROR: If multithreading should be used, the input files have to be specified as cmd arguments.")
        sys.exit(-1)

    import multiprocessing
    n_cores = multiprocessing.cpu_count()

    if n_threads > n_cores:
        print("Requested number of threads exceeds number of available cpu cores. N threads will be set to %i." % n_cores)
        n_threads = n_cores

    if n_threads > len(args["data_files"]):
        print("More parallel jobs then input files specified. Limiting the maximum number of input files to %i" % len(args["data_files"]))
        n_threads = len(args["data_files"])


print("outputpath: ", output_path)

# create bootstrap_uuid.xml
parser = ET.XMLParser(remove_blank_text=True, remove_comments=False, attribute_defaults=False, recover=True, resolve_entities=False)
BootstrapTree = ET.parse(bootstrap_path, parser)
BootstrapRoot = BootstrapTree.getroot()
docinfo = BootstrapTree.docinfo
offline_config_path = BootstrapRoot.get(XSI + "noNamespaceSchemaLocation")[:-14]

# convert paths for all configLink sepcified with type=XML (but not EventFileReader, ModuleSequence)
for elem in BootstrapRoot.findall(".//configLink[@type='XML']"):
    if elem.get("id") == "EventFileReader" or elem.get("id") == "ModuleSequence":
        continue

    if bootstrap_dir != "./":
        print("Change Path of %s.xml to: %s%s.xml" % (elem.get("id"), bootstrap_dir, elem.get("id")))
        elem.set(XLINK + "href", elem.get(XLINK + "href").replace("./", bootstrap_dir))

# get count data_files in EventFileReader
if args['data_files'] is None:
    config_link_modulesequence = BootstrapRoot.find(".//configLink[@id='EventFileReader']")
    current_path = config_link_modulesequence.get(XLINK + "href").replace("./", bootstrap_dir)
    # Remove Comments needs to be set if comments are in the InputFilenames list
    parser = ET.XMLParser(remove_blank_text=True, remove_comments=True, attribute_defaults=False, recover=True, resolve_entities=False)
    event_file_tree = ET.parse(current_path, parser)
    event_file_root = event_file_tree.getroot()
    filenames_tag = event_file_root.find(".//InputFilenames")
    data_files = []
    for line in filenames_tag.text.split("\n"):
        if not re.match(r'^\s*$', line):  # skip empty lines
            data_files.append(line.strip())  # remove trailing whitespaces
else:
    data_files = args['data_files']


# create Eventfilereader.xml only if data files have been specified, otherwise use link of bootstrap file
imin, imax = 0, 0
# n_additional_files = len(data_files) - int(len(data_files) / n_threads) * n_threads
n_additional_files = len(data_files) % n_threads

jobs = []
N = len(data_files) // n_threads  # needs to be integer devision
# main loop to prepare jobs
for i in range(n_threads):
    pattern = ""
    pattern += "_%i" % i if n_threads > 1 else ""
    filename_pattern = str(_uuid) + pattern

    # define range of files per job
    imax = imin + N - 1
    if n_additional_files > 0:
        imax += 1
        n_additional_files -= 1

    if(i == n_threads - 1):
        imax = len(data_files)

    if args['data_files'] is not None:
        ''' create EventFileReader.xml '''
        eventFileReaderRoot = ET.fromstring("""
            <EventFileReader xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                xsi:noNamespaceSchemaLocation="%s">
                <%s>
                    <InputFilenames></InputFilenames>
                </%s>
            </EventFileReader>""" % (os.path.join(offline_config_path, 'EventFileReader.xsd'), data_file_type, data_file_type))
        EventFileReaderTree = eventFileReaderRoot.getroottree()

        filenames_tag = eventFileReaderRoot.find(".//InputFilenames")
        for filename in data_files[imin:imax]:
            if(filenames_tag.text is None):
                filenames_tag.text = "\n\t" + filename + "\n"
            else:
                filenames_tag.text = filenames_tag.text + "\t" + filename + "\n"

        docinfo = EventFileReaderTree.docinfo
        with open(os.path.join(output_path, filename_pattern + "_EventFileReader.xml"), "w") as f:
            f.write(ET.tostring(EventFileReaderTree, pretty_print=True, xml_declaration=True, encoding=docinfo.encoding).decode())

        # create config link if it does not exists
        config_link_event = BootstrapRoot.find(".//configLink[@id='EventFileReader']")
        if config_link_event is None:
            config_link_event = ET.SubElement(BootstrapRoot.find("./centralConfig"), "configLink")
            config_link_event.set("id", "EventFileReader")
            config_link_event.set("type", "XML")

        config_link_event.set(XLINK + "href", os.path.join(output_path, filename_pattern + "_EventFileReader.xml"))
    else:
        # copy eventfilereader to output directory and modify link in bootstrap
        eventfilereader_path = os.path.join(output_path, filename_pattern + "_EventFileReader.xml")
        config_link_eventfilereader = BootstrapRoot.find(".//configLink[@id='EventFileReader']")
        current_path = config_link_eventfilereader.get(XLINK + "href").replace("./", bootstrap_dir)
        shutil.copyfile(current_path, eventfilereader_path)  # copy model sequence into output_path
        config_link_eventfilereader.set(XLINK + "href", eventfilereader_path)

    # change output path of RecDataWriter
    output_file_name = os.path.join(adst_output_path, filename_pattern + "_" + args['output_file_name'] + ".root")
    setModuleOptions(BootstrapRoot, "RecDataWriter", [["outputFileName", output_file_name, None, ["rootOutput"]]])

    # set module options
    if len(module_options):
        for j in xrange(len(module_options)):
            # add indices for module options
            current_module_option = copy.copy(module_options[j])
            if(module_options[j][1].find("{i}") != -1):
                current_module_option[1] = current_module_option[1].replace("{i}", str(i))

            setModuleOptions(BootstrapRoot, module_names[j], [current_module_option])

    # copy modulesequence to output directory and modify link in bootstrap
    modulesequence_path = os.path.join(output_path, filename_pattern + "_ModuleSequence.xml")
    config_link_modulesequence = BootstrapRoot.find(".//configLink[@id='ModuleSequence']")
    current_path = config_link_modulesequence.get(XLINK + "href").replace("./", bootstrap_dir)
    shutil.copyfile(current_path, modulesequence_path)  # copy model sequence into output_path
    config_link_modulesequence.set(XLINK + "href", modulesequence_path)

    # create bootstrap file
    bootstrap_string = getFormattedBootstrap(ET.tostring(BootstrapTree, pretty_print=True, xml_declaration=True, encoding=docinfo.encoding).decode())
    with open(os.path.join(output_path, filename_pattern + "_bootstrap.xml"), "w") as f:
        f.write(bootstrap_string.decode())

    fstdout_name = os.path.join(output_path, filename_pattern + ".out")
    fstdout = open(fstdout_name, "w")

    cmd = user_auger_offline + " -b " + os.path.join(output_path, filename_pattern + "_bootstrap.xml")
    print("executing command %s\n\tlog will be written to: %s\n\tADST file will be written to %s, " % (cmd, os.path.join(output_path, filename_pattern + ".out"), os.path.join(output_path, output_file_name)))

    jobs.append(subprocess.Popen(shlex.split(cmd), stdout=fstdout, stderr=subprocess.STDOUT))

    # set starting file for next job
    imin = imax + 1


t_start = time.time()
thread_range = np.arange(n_threads)


def cleanup():
    for idx in thread_range:
        jobs[idx].kill()


atexit.register(cleanup)

exitcodes = []
while True:
    # Once all jobs are done
    if(len(thread_range) == 0):
        break

    time.sleep(1)
    sys.stdout.write("\rrunning... %s" % (str(datetime.timedelta(seconds=int(time.time() - t_start)))))
    sys.stdout.flush()

    for idx in thread_range:
        if not jobs[idx].poll() is None:
            exitcode = jobs[idx].poll()
            exitcodes.append(exitcode)
            print("\rJob %i finished with exit code %i (t = %s)\n" % (idx, exitcode, str(datetime.timedelta(seconds=int(time.time() - t_start)))))

            # after job is finished remove it from the list
            index = np.arange(len(thread_range))[thread_range == idx]
            thread_range = np.delete(thread_range, index)

# finishes the program with the lowest exitcode
sys.exit(min(exitcodes))
