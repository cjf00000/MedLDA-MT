import json

def read_log(file):
    with open(file) as f:
        lines = []
        for line in f:
            if line.find('iteration') != -1:
                line = line.replace(", }", "}")
                lines.append(json.loads(line))
    return lines