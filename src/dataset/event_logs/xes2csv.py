import xmltodict
import sys
import csv
from collections import OrderedDict

filename = sys.argv[1] 
outfile = sys.argv[2]

with open(filename, 'r') as f:
    doc = xmltodict.parse(f.read())

def get_properties(trace):
    trace_info = {}
    for k, v in trace.items():
        if k != 'event':
            if not isinstance(v, OrderedDict):
                for att in v:
                    trace_info[att['@key']] = att['@value']
            else:
                trace_info[v['@key']] = v['@value']
    return trace_info

traces = doc['log']['trace']
print(traces[0].keys())
with open(outfile, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Case ID','Activity', 'Complete Timestamp'])
    for trace in traces:
        #print(trace)
        if not isinstance(trace, OrderedDict):
            continue
        caseid = get_properties(trace)['concept:name']  
        for event in trace['event']:
            if not isinstance(event, OrderedDict):
                continue
            props = get_properties(event)
            writer.writerow([caseid, props['concept:name'], props['time:timestamp']])