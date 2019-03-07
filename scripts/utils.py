import json

def read_log(file):
    with open(file) as f:
        lines = []
        for line in f:
            if line.find('iteration') != -1:
                line = line.replace(", }", "}")
                lines.append(json.loads(line))
    return lines

def read_light_log(file):
    with open(file) as f:
        lines = []
        for line in f:
            if line.find('Test Accuracy') != -1:
                lines.append(float(line.split()[-2]))
    return lines